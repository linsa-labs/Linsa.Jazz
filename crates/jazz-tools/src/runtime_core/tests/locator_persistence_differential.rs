//! v18 items 4/5: the differential oracle over locator healing (methodology step 9).
//!
//! `locator_warmth.rs` gates the shapes we thought of: one read, one restart, one pass. The D2
//! hook's correctness is an ORDERING property — the recovery is written inside whatever
//! transaction happens to be open, and which one that is depends on whether a pass is running,
//! whether a barrier has run since, and whether the store was reopened in between. This runs
//! randomised interleavings of those and checks the invariant after every step.
//!
//! Internal on purpose: the observable is `visible_ladder_recoveries_for_test()`, a per-store
//! count of a walk that happens entirely inside one storage call. No client API exposes it —
//! a client sees the row either way, which is exactly why the defect ran for months.

use super::support::{docs_v2, runtime_over, split_store};
use super::*;
use crate::query_manager::manager::LocalUpdates;
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryPropagation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// Read the split row on branch A — the operation that can walk the ladder.
    Read,
    /// Run a batched tick, which runs the durability barrier.
    Barrier,
    /// Close the store and reopen it over the same file.
    Restart,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn read_branch(core: &mut RuntimeCore<SqliteStorage, NoopScheduler>, branch: &BranchName) {
    let (_handle, _future) = core
        .query_with_local_batch_tracked(
            QueryBuilder::new("docs").branch(branch.as_str()).build(),
            None,
            ReadDurabilityOptions {
                tier: None,
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::LocalOnly,
            None,
        )
        .expect("query setup");
}

/// The invariant, in one sentence: a row healed and then barriered never walks again.
///
/// The subtlety the differential exists for is the pair (heal, barrier). A heal that has NOT
/// been through a barrier may legitimately be lost to a restart — the transaction it was
/// written in was never committed — so the model tracks `healed` and `durable` separately, and
/// only `durable` survives a `Restart`. Anything stricter would be a wrong oracle; anything
/// looser would not catch a D2 hook that writes outside the pass transaction.
#[test]
fn a_healed_locator_never_walks_again_under_random_interleavings() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    for seed in 1..=40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let path = dir.path().join(format!("locdiff-{seed}.sqlite"));
        let _ = std::fs::remove_file(&path);

        let (mut core, row_id, branch_a) = split_store(&path, "loc-differential");
        let _ = row_id;

        // Model state.
        let mut healed = false; // the pointer is in the store, committed or not
        let mut durable = false; // ... and a barrier has committed it
        let mut program: Vec<Op> = Vec::new();

        let steps = 4 + rng.below(9);
        for _ in 0..steps {
            let op = match rng.below(3) {
                0 => Op::Read,
                1 => Op::Barrier,
                _ => Op::Restart,
            };
            program.push(op);

            match op {
                Op::Read => {
                    let before = core.storage().visible_ladder_recoveries_for_test();
                    read_branch(&mut core, &branch_a);
                    let walked = core.storage().visible_ladder_recoveries_for_test() > before;
                    if healed {
                        assert!(
                            !walked,
                            "seed {seed} program {program:?}: a row whose exact locator was \
                             already recovered must be answered from the pointer; walking the \
                             families again is the 1,200-walks-in-five-minutes defect"
                        );
                    } else {
                        assert!(
                            walked,
                            "seed {seed} program {program:?}: the first read of a split row \
                             must walk the ladder — if it does not, the fixture stopped \
                             reproducing the split and every later assertion is vacuous"
                        );
                        healed = true;
                    }
                }
                Op::Barrier => {
                    core.batched_tick();
                    if healed {
                        durable = true;
                    }
                    assert!(
                        core.storage().is_autocommit_for_test(),
                        "seed {seed} program {program:?}: the barrier must leave no \
                         transaction open"
                    );
                }
                Op::Restart => {
                    let storage = core.into_storage();
                    drop(storage);
                    let storage = SqliteStorage::open(&path).expect("reopen");
                    core = runtime_over(docs_v2(), "loc-differential", storage);
                    // A heal that never reached a barrier was in an uncommitted transaction.
                    healed = durable;
                }
            }
        }
    }
}
