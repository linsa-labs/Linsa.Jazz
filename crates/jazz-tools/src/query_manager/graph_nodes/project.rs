use ahash::AHashSet;

use crate::object::ObjectId;
use crate::query_manager::encoding::{decode_column, encode_row};
use crate::query_manager::graph_nodes::tuple_delta::{
    compute_tuple_delta, has_tuple_content_changed,
};
use crate::query_manager::relation_ir::{ProjectColumn, ProjectExpr, RowIdRef};
use crate::query_manager::types::{
    ColumnDescriptor, ColumnName, ColumnType, RowDescriptor, Tuple, TupleDelta, TupleDescriptor,
    TupleElement, Value,
};

use super::RowNode;

#[derive(Debug, Clone)]
enum ProjectionSource {
    Column { global_index: usize },
    RowId { element_index: usize },
}

#[derive(Debug, Clone)]
struct ProjectionField {
    output_column: ColumnDescriptor,
    source: ProjectionSource,
}

/// Project node for column selection.
///
/// Transforms fully materialized tuples into a single output row with the
/// requested projection shape. This supports both the legacy "select these
/// column names" path and precise relation-IR projections with aliases/scopes.
#[derive(Debug)]
pub struct ProjectNode {
    input_tuple_descriptor: TupleDescriptor,
    output_descriptor: RowDescriptor,
    output_tuple_descriptor: TupleDescriptor,
    projection_fields: Vec<ProjectionField>,
    current_tuples: AHashSet<Tuple>,
    ordered_tuples: Vec<Tuple>,
    /// The ids of the ordered input the list was last rebuilt from, flattened, with the
    /// tuples' arity: what an in-place update checks the next ordered input against.
    /// `None` when some input tuple did not project, so a position in the input is not
    /// a position in the list.
    in_place_ids: Option<(usize, Vec<ObjectId>)>,
    dirty: bool,
    /// Passes this node took the in-place path in, for tests that must tell it from an
    /// include instance's.
    #[cfg(test)]
    pub(crate) applied_in_place: usize,
}

impl ProjectNode {
    /// Create a new project node from a single input row descriptor.
    pub fn new(input_descriptor: RowDescriptor, select_columns: &[&str]) -> Self {
        let input_tuple_descriptor =
            TupleDescriptor::single_with_materialization("", input_descriptor, true);
        Self::with_tuple_descriptor(input_tuple_descriptor, select_columns)
    }

    /// Create a new project node from a tuple descriptor and unqualified column names.
    pub fn with_tuple_descriptor(
        input_tuple_descriptor: TupleDescriptor,
        select_columns: &[&str],
    ) -> Self {
        let combined_descriptor = input_tuple_descriptor.combined_descriptor();
        let projection_fields = select_columns
            .iter()
            .filter_map(|col_name| {
                let src_idx = combined_descriptor.column_index(col_name)?;
                let col = combined_descriptor.columns[src_idx].clone();
                Some(ProjectionField {
                    output_column: col,
                    source: ProjectionSource::Column {
                        global_index: src_idx,
                    },
                })
            })
            .collect();

        Self::from_projection_fields(input_tuple_descriptor, projection_fields)
    }

    /// Create a new project node from explicit projected expressions.
    pub fn with_project_columns(
        input_tuple_descriptor: TupleDescriptor,
        project_columns: &[ProjectColumn],
    ) -> Option<Self> {
        let mut projection_fields = Vec::with_capacity(project_columns.len());
        for column in project_columns {
            let Some(source) = (match &column.expr {
                ProjectExpr::Column(column_ref) if column_ref.column == "_id" => {
                    Some(ProjectionSource::RowId {
                        element_index: resolve_row_id_element(
                            &input_tuple_descriptor,
                            column_ref.scope.as_deref(),
                        )?,
                    })
                }
                ProjectExpr::Column(column_ref) => {
                    let global_index = if let Some(scope) = column_ref.scope.as_deref() {
                        input_tuple_descriptor.qualified_column_index(scope, &column_ref.column)
                    } else {
                        input_tuple_descriptor.column_index(&column_ref.column)
                    };
                    global_index.map(|global_index| ProjectionSource::Column { global_index })
                }
                ProjectExpr::RowId(RowIdRef::Current) => {
                    resolve_row_id_element(&input_tuple_descriptor, None)
                        .map(|element_index| ProjectionSource::RowId { element_index })
                }
                ProjectExpr::RowId(_) => None,
            }) else {
                continue;
            };

            let output_column = match &source {
                ProjectionSource::Column { global_index } => {
                    let (element_index, local_index) =
                        input_tuple_descriptor.resolve_column(*global_index)?;
                    let source_column = &input_tuple_descriptor
                        .element(element_index)?
                        .descriptor
                        .columns[local_index];
                    ColumnDescriptor {
                        name: ColumnName::new(column.alias.clone()),
                        column_type: source_column.column_type.clone(),
                        nullable: source_column.nullable,
                        references: source_column.references,
                        default: source_column.default.clone(),
                        merge_strategy: source_column.merge_strategy,
                    }
                }
                ProjectionSource::RowId { .. } => ColumnDescriptor {
                    name: ColumnName::new(column.alias.clone()),
                    column_type: ColumnType::Uuid,
                    nullable: false,
                    references: None,
                    default: None,
                    merge_strategy: None,
                },
            };

            projection_fields.push(ProjectionField {
                output_column,
                source,
            });
        }

        if projection_fields.is_empty() && !project_columns.is_empty() {
            return None;
        }

        Some(Self::from_projection_fields(
            input_tuple_descriptor,
            projection_fields,
        ))
    }

    fn from_projection_fields(
        input_tuple_descriptor: TupleDescriptor,
        projection_fields: Vec<ProjectionField>,
    ) -> Self {
        let output_descriptor = RowDescriptor::new(
            projection_fields
                .iter()
                .map(|field| field.output_column.clone())
                .collect::<Vec<_>>(),
        );
        let output_tuple_descriptor =
            TupleDescriptor::single_with_materialization("", output_descriptor.clone(), true);

        Self {
            input_tuple_descriptor,
            output_descriptor,
            output_tuple_descriptor,
            projection_fields,
            current_tuples: AHashSet::new(),
            ordered_tuples: Vec::new(),
            in_place_ids: None,
            dirty: true,
            #[cfg(test)]
            applied_in_place: 0,
        }
    }

    /// Get the output tuple descriptor.
    pub fn output_tuple_descriptor(&self) -> &TupleDescriptor {
        &self.output_tuple_descriptor
    }

    /// Projected tuples in the same order as an ordered upstream node.
    pub fn ordered_tuples(&self) -> &[Tuple] {
        &self.ordered_tuples
    }

    /// Process an update while preserving the exact order supplied by an
    /// upstream sort, pagination, or ordered projection node.
    pub fn process_with_ordered_input(
        &mut self,
        input: TupleDelta,
        ordered_input: &[Tuple],
    ) -> TupleDelta {
        #[cfg(test)]
        crate::query_manager::ordered_in_place::note_rebuilt_tuples(ordered_input.len());
        let old_ordered = std::mem::take(&mut self.ordered_tuples);
        RowNode::process(self, input);
        self.ordered_tuples = ordered_input
            .iter()
            .filter_map(|tuple| self.project_tuple(tuple))
            .collect();
        self.current_tuples = self.ordered_tuples.iter().cloned().collect();
        // A list that holds one id twice cannot be updated in place: the rebuild reports the
        // duplicate as moved on every pass (`compute_tuple_delta` keeps the last position of an
        // id), and a projection that keeps only the first id makes one from a 1:N join. The set
        // is keyed by ids, so it is shorter than the list exactly then.
        let arity = ordered_input.first().map_or(0, Tuple::len);
        self.in_place_ids = (arity > 0
            && self.ordered_tuples.len() == ordered_input.len()
            && self.current_tuples.len() == self.ordered_tuples.len()
            && ordered_input.iter().all(|tuple| tuple.len() == arity))
        .then(|| {
            (
                arity,
                ordered_input
                    .iter()
                    .flat_map(|tuple| tuple.id_iter())
                    .collect(),
            )
        });
        compute_tuple_delta(&old_ordered, &self.ordered_tuples)
    }

    /// The length of the ordered input the list was built from, or `None` when the list
    /// cannot be updated in place at all.
    pub(crate) fn in_place_len(&self) -> Option<usize> {
        let (arity, ids) = self.in_place_ids.as_ref()?;
        Some(ids.len() / arity)
    }

    /// Whether `tuple`, at `position` of the ordered input, has the ids the list was built
    /// from there.
    pub(crate) fn in_place_ids_match(&self, position: usize, tuple: &Tuple) -> bool {
        let Some((arity, ids)) = self.in_place_ids.as_ref() else {
            return false;
        };
        tuple.len() == *arity
            && ids
                .get(position * arity..(position + 1) * arity)
                .is_some_and(|expected| tuple.id_iter().eq(expected.iter().copied()))
    }

    /// The projections of `updates` (positions in the ordered input, which are positions
    /// in the list), or `None` when one does not project.
    pub(crate) fn project_in_place(
        &self,
        updates: &[(usize, Tuple)],
    ) -> Option<Vec<(usize, Tuple)>> {
        updates
            .iter()
            .map(|(position, tuple)| Some((*position, self.project_tuple(tuple)?)))
            .collect()
    }

    /// The update-only pass of `process_with_ordered_input` when the ordered input kept
    /// its ids: replace the projected tuples where they stand.
    pub(crate) fn apply_in_place(&mut self, projected: Vec<(usize, Tuple)>) -> TupleDelta {
        #[cfg(test)]
        {
            self.applied_in_place += 1;
        }
        let mut delta = TupleDelta::new();
        for (position, new) in projected {
            let old = std::mem::replace(&mut self.ordered_tuples[position], new.clone());
            self.current_tuples.replace(new.clone());
            if has_tuple_content_changed(&old, &new) {
                delta.updated.push((old, new));
            }
        }
        self.dirty = false;
        delta
    }

    fn projected_value(&self, tuple: &Tuple, source: &ProjectionSource) -> Option<Value> {
        match source {
            ProjectionSource::Column { global_index } => {
                let (element_index, local_index) =
                    self.input_tuple_descriptor.resolve_column(*global_index)?;
                let element = tuple.get(element_index)?;
                let descriptor = &self
                    .input_tuple_descriptor
                    .element(element_index)?
                    .descriptor;
                decode_column(descriptor, element.content()?, local_index).ok()
            }
            ProjectionSource::RowId { element_index } => {
                Some(Value::Uuid(tuple.get(*element_index)?.id()))
            }
        }
    }

    /// Project a single tuple to the output row shape.
    fn project_tuple(&self, tuple: &Tuple) -> Option<Tuple> {
        let values: Option<Vec<_>> = self
            .projection_fields
            .iter()
            .map(|field| self.projected_value(tuple, &field.source))
            .collect();
        let projected_content = encode_row(&self.output_descriptor, &values?).ok()?;
        let id = tuple.first_id()?;
        let batch_id = tuple
            .iter()
            .find_map(TupleElement::batch_id)
            .unwrap_or(crate::row_histories::BatchId([0; 16]));
        let row_provenance = tuple
            .iter()
            .find_map(TupleElement::row_provenance)
            .cloned()?;

        Some(
            Tuple::new(vec![TupleElement::Row {
                id,
                content: projected_content.into(),
                batch_id,
                row_provenance,
            }])
            .with_provenance(tuple.provenance().clone())
            .with_batch_provenance(tuple.batch_provenance().clone()),
        )
    }
}

fn resolve_row_id_element(
    input_tuple_descriptor: &TupleDescriptor,
    scope: Option<&str>,
) -> Option<usize> {
    if let Some(scope) = scope {
        for index in 0..input_tuple_descriptor.element_count() {
            if input_tuple_descriptor.element(index)?.table == scope {
                return Some(index);
            }
        }
        return None;
    }

    (input_tuple_descriptor.element_count() == 1).then_some(0)
}

impl RowNode for ProjectNode {
    fn output_descriptor(&self) -> &RowDescriptor {
        &self.output_descriptor
    }

    fn process(&mut self, input: TupleDelta) -> TupleDelta {
        let mut result = TupleDelta::new();
        self.in_place_ids = None;

        for tuple in input.removed {
            if let Some(projected) = self.project_tuple(&tuple) {
                self.current_tuples.remove(&projected);
                result.removed.push(projected);
            }
        }

        for tuple in input.added {
            if let Some(projected) = self.project_tuple(&tuple) {
                self.current_tuples.insert(projected.clone());
                result.added.push(projected);
            }
        }

        for (old_tuple, new_tuple) in input.updated {
            if let (Some(old_projected), Some(new_projected)) = (
                self.project_tuple(&old_tuple),
                self.project_tuple(&new_tuple),
            ) {
                self.current_tuples.remove(&old_projected);
                self.current_tuples.insert(new_projected.clone());
                result.updated.push((old_projected, new_projected));
            }
        }

        self.dirty = false;
        result
    }

    fn current_tuples(&self) -> &AHashSet<Tuple> {
        &self.current_tuples
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn is_dirty(&self) -> bool {
        self.dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectId;
    use crate::query_manager::encoding::{decode_row, encode_row};
    use crate::query_manager::relation_ir::ColumnRef;
    use crate::query_manager::types::Value;

    fn test_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("name", ColumnType::Text),
            ColumnDescriptor::new("email", ColumnType::Text),
            ColumnDescriptor::new("age", ColumnType::Integer),
        ])
    }

    fn make_tuple(id: ObjectId, values: &[Value]) -> Tuple {
        let descriptor = test_descriptor();
        let data = encode_row(&descriptor, values).unwrap();
        Tuple::new(vec![TupleElement::Row {
            id,
            content: data.into(),
            batch_id: crate::row_histories::BatchId([0; 16]),
            row_provenance: crate::metadata::RowProvenance::for_insert("jazz:test", 0),
        }])
    }

    #[test]
    fn project_selects_columns() {
        let descriptor = test_descriptor();
        let mut node = ProjectNode::new(descriptor, &["name", "age"]);

        let id1 = ObjectId::new();
        let tuple1 = make_tuple(
            id1,
            &[
                Value::Integer(1),
                Value::Text("Alice".into()),
                Value::Text("alice@example.com".into()),
                Value::Integer(30),
            ],
        );

        let delta = TupleDelta {
            added: vec![tuple1],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };

        let result = node.process(delta);

        assert_eq!(result.added.len(), 1);

        let projected = &result.added[0];
        let row = projected.to_single_row().unwrap();
        let values = decode_row(node.output_descriptor(), &row.data).unwrap();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], Value::Text("Alice".into()));
        assert_eq!(values[1], Value::Integer(30));
    }

    #[test]
    fn project_output_descriptor_has_selected_columns() {
        let descriptor = test_descriptor();
        let node = ProjectNode::new(descriptor, &["name", "age"]);

        let output = node.output_descriptor();
        assert_eq!(output.columns.len(), 2);
        assert_eq!(output.columns[0].name, "name");
        assert_eq!(output.columns[1].name, "age");
    }

    #[test]
    fn project_preserves_tuple_identity() {
        let descriptor = test_descriptor();
        let mut node = ProjectNode::new(descriptor, &["name"]);

        let id1 = ObjectId::new();
        let tuple1 = make_tuple(
            id1,
            &[
                Value::Integer(1),
                Value::Text("Alice".into()),
                Value::Text("alice@example.com".into()),
                Value::Integer(30),
            ],
        );

        let delta = TupleDelta {
            added: vec![tuple1],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };

        node.process(delta);

        assert_eq!(node.current_tuples().len(), 1);
        let projected = node.current_tuples().iter().next().unwrap();
        assert_eq!(projected.ids()[0], id1);
    }

    #[test]
    fn project_handles_updates() {
        let descriptor = test_descriptor();
        let mut node = ProjectNode::new(descriptor, &["name", "age"]);

        let id1 = ObjectId::new();
        let old_tuple = make_tuple(
            id1,
            &[
                Value::Integer(1),
                Value::Text("Alice".into()),
                Value::Text("alice@example.com".into()),
                Value::Integer(30),
            ],
        );
        let new_tuple = make_tuple(
            id1,
            &[
                Value::Integer(1),
                Value::Text("Alice".into()),
                Value::Text("alice@newmail.com".into()),
                Value::Integer(31),
            ],
        );

        node.process(TupleDelta {
            added: vec![old_tuple.clone()],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        });

        let result = node.process(TupleDelta {
            added: vec![],
            removed: vec![],
            moved: vec![],
            updated: vec![(old_tuple, new_tuple)],
        });

        assert_eq!(result.updated.len(), 1);

        let (_, new_projected) = &result.updated[0];
        let row = new_projected.to_single_row().unwrap();
        let values = decode_row(node.output_descriptor(), &row.data).unwrap();
        assert_eq!(values[1], Value::Integer(31));
    }

    // Scenario: an in-place pass reports only the tuples whose projection changed; an input
    // change in a column the projection drops is not an update of the projected list.
    #[test]
    fn in_place_reports_only_changed_projections() {
        let mut node = ProjectNode::new(test_descriptor(), &["name", "age"]);
        let (id1, id2) = (ObjectId::new(), ObjectId::new());
        let row = |id, email: &str, age| {
            make_tuple(
                id,
                &[
                    Value::Integer(1),
                    Value::Text("Alice".into()),
                    Value::Text(email.into()),
                    Value::Integer(age),
                ],
            )
        };
        let before = [row(id1, "a@example.com", 30), row(id2, "b@example.com", 40)];
        node.process_with_ordered_input(
            TupleDelta {
                added: before.to_vec(),
                ..TupleDelta::new()
            },
            &before,
        );
        assert_eq!(node.in_place_len(), Some(2));

        let updates = [
            (0, row(id1, "new@example.com", 30)),
            (1, row(id2, "b@example.com", 41)),
        ];
        let projected = node.project_in_place(&updates).expect("both rows project");
        let delta = node.apply_in_place(projected);

        let updated: Vec<_> = delta
            .updated
            .iter()
            .map(|(_, new)| new.first_id())
            .collect();
        assert_eq!(
            updated,
            vec![Some(id2)],
            "an unprojected column change is not an update"
        );
        let values = |tuple: &Tuple| {
            decode_row(
                node.output_descriptor(),
                &tuple.to_single_row().unwrap().data,
            )
            .unwrap()
        };
        assert_eq!(values(&node.ordered_tuples()[1])[1], Value::Integer(41));
        let held = node
            .current_tuples()
            .get(&node.ordered_tuples()[1])
            .expect("held");
        assert_eq!(
            values(held)[1],
            Value::Integer(41),
            "the unordered view holds the new projection"
        );
    }

    #[test]
    fn project_ignores_unknown_columns() {
        let descriptor = test_descriptor();
        let node = ProjectNode::new(descriptor, &["name", "nonexistent", "age"]);

        let output = node.output_descriptor();
        assert_eq!(output.columns.len(), 2);
        assert_eq!(output.columns[0].name, "name");
        assert_eq!(output.columns[1].name, "age");
    }

    // Scenario: a list that holds one id twice is never updated in place: every rebuild
    // reports the duplicate as moved. A projection that keeps only the first id of a 1:N join
    // makes one; here the input itself repeats an id.
    #[test]
    fn a_projected_list_holding_one_id_twice_is_never_updated_in_place() {
        let mut node = ProjectNode::new(test_descriptor(), &["name"]);
        let (a, b) = (ObjectId::new(), ObjectId::new());
        let row = |id, name: &str| {
            make_tuple(
                id,
                &[
                    Value::Integer(1),
                    Value::Text(name.into()),
                    Value::Text("x@example.com".into()),
                    Value::Integer(30),
                ],
            )
        };
        node.process_with_ordered_input(TupleDelta::default(), &[row(a, "Alice"), row(b, "Bob")]);
        assert_eq!(node.in_place_len(), Some(2), "fixture: distinct ids can");

        node.process_with_ordered_input(
            TupleDelta::default(),
            &[row(a, "Alice"), row(a, "Alicia"), row(b, "Bob")],
        );

        assert_eq!(node.in_place_len(), None);
    }

    // Scenario: an in-place pass over a tuple whose projected data is as it was but whose
    // provenance or batch provenance changed reports it: the Output after it moves its scope
    // on that update, and the client reads the batch provenance.
    #[test]
    fn in_place_reports_provenance_and_batch_only_changes() {
        let mut node = ProjectNode::new(test_descriptor(), &["name"]);
        let id = ObjectId::new();
        let row = || {
            make_tuple(
                id,
                &[
                    Value::Integer(1),
                    Value::Text("Alice".into()),
                    Value::Text("x@example.com".into()),
                    Value::Integer(30),
                ],
            )
        };
        node.process_with_ordered_input(TupleDelta::default(), &[row()]);
        let projected = node.ordered_tuples()[0].clone();
        let b = crate::object::BranchName::new("b");

        let moved = projected
            .clone()
            .with_provenance([(id, b)].into_iter().collect());
        let delta = node.apply_in_place(vec![(0, moved)]);
        assert_eq!(
            delta.updated.len(),
            1,
            "a provenance-only change is an update"
        );

        let rebatched = node.ordered_tuples()[0].clone().with_batch_provenance(
            [crate::row_histories::BatchId([7; 16])]
                .into_iter()
                .collect(),
        );
        let delta = node.apply_in_place(vec![(0, rebatched)]);
        assert_eq!(delta.updated.len(), 1, "a batch-only change is an update");
    }

    #[test]
    fn precise_project_uses_scopes_and_aliases() {
        let users = RowDescriptor::new(vec![ColumnDescriptor::new("name", ColumnType::Text)]);
        let posts = RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]);
        let input = TupleDescriptor::from_tables(&[
            ("u".to_string(), users.clone()),
            ("p".to_string(), posts.clone()),
        ])
        .with_all_materialized();
        let node = ProjectNode::with_project_columns(
            input,
            &[
                ProjectColumn {
                    alias: "author_name".into(),
                    expr: ProjectExpr::Column(ColumnRef::scoped("u", "name")),
                },
                ProjectColumn {
                    alias: "post_title".into(),
                    expr: ProjectExpr::Column(ColumnRef::scoped("p", "title")),
                },
            ],
        )
        .expect("precise projection should build");

        assert_eq!(node.output_descriptor().columns.len(), 2);
        assert_eq!(node.output_descriptor().columns[0].name, "author_name");
        assert_eq!(node.output_descriptor().columns[1].name, "post_title");

        let user_id = ObjectId::new();
        let post_id = ObjectId::new();
        let user_row = encode_row(&users, &[Value::Text("Alice".into())]).unwrap();
        let post_row = encode_row(&posts, &[Value::Text("Hello".into())]).unwrap();
        let tuple = Tuple::new(vec![
            TupleElement::Row {
                id: user_id,
                content: user_row.into(),
                batch_id: crate::row_histories::BatchId([1; 16]),
                row_provenance: crate::metadata::RowProvenance::for_insert("jazz:test", 0),
            },
            TupleElement::Row {
                id: post_id,
                content: post_row.into(),
                batch_id: crate::row_histories::BatchId([2; 16]),
                row_provenance: crate::metadata::RowProvenance::for_insert("jazz:test", 0),
            },
        ]);

        let result = node
            .project_tuple(&tuple)
            .expect("tuple should project to a single row");
        let row = result
            .to_single_row()
            .expect("projected tuple should be a row");
        let values = decode_row(node.output_descriptor(), &row.data).unwrap();

        assert_eq!(row.id, user_id);
        assert_eq!(
            values,
            vec![Value::Text("Alice".into()), Value::Text("Hello".into())]
        );
    }

    #[test]
    fn precise_project_can_expose_scoped_row_id() {
        let users = RowDescriptor::new(vec![ColumnDescriptor::new("name", ColumnType::Text)]);
        let input =
            TupleDescriptor::from_tables(&[("u".to_string(), users)]).with_all_materialized();
        let node = ProjectNode::with_project_columns(
            input,
            &[ProjectColumn {
                alias: "row_id".into(),
                expr: ProjectExpr::Column(ColumnRef::scoped("u", "_id")),
            }],
        )
        .expect("row id projection should build");

        assert_eq!(
            node.output_descriptor().columns[0].column_type,
            ColumnType::Uuid
        );
        assert_eq!(node.output_descriptor().columns[0].references, None);
        assert_eq!(node.output_descriptor().columns[0].name, "row_id");
    }
}
