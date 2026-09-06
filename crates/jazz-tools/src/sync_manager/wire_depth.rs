//! A nesting bound for everything decoded off the wire.
//!
//! Postcard has no recursion limit, and every recursive wire type — `PredicateExpr::Not`,
//! the boxed `RelExpr` inputs, nested include specs, `Value::Array` — is a derived,
//! recursive deserializer. A frame of ~12 KB of nested `Not` overflows a tokio worker's
//! 2 MiB stack at decode, which is a `SIGABRT` for the whole process, before any admission
//! check can see the frame. The bound lives at the decode, not on the types: wrapping the
//! deserializer counts every level of nesting (sequence, map, enum, option, newtype) and
//! refuses past `WIRE_MAX_NESTING`, so no recursive type needs to know about it and none can
//! be forgotten.
//!
//! The bound is a memory-safety limit, not a policy: it is a constant, about 2.8× above
//! the deepest shape the query builders produce, and equal to `serde_json`'s default
//! recursion limit. As the adapter counts (one level per sequence, map, enum, option or
//! newtype — two per relation node plus one per container), eight joins under limit,
//! offset, order-by and projection with a recursive spec beneath them nest around 45
//! levels; a six-level include tree about 23. A frame at the bound decodes on a 2 MiB
//! worker stack (`tests/admission.rs`, `a_frame_at_the_wire_bound_decodes_on_a_worker_stack`).

use std::cell::Cell;

use serde::de::{
    DeserializeSeed, Deserializer, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor,
};

/// Deepest nesting accepted from the wire.
pub const WIRE_MAX_NESTING: usize = 128;

/// `postcard::from_bytes` with the nesting bound applied.
pub fn from_postcard_bounded<'de, T: serde::Deserialize<'de>>(
    bytes: &'de [u8],
    max_nesting: usize,
) -> postcard::Result<T> {
    let depth = Cell::new(0);
    let mut inner = postcard::Deserializer::from_bytes(bytes);
    T::deserialize(Bounded {
        inner: &mut inner,
        depth: &depth,
        max: max_nesting,
    })
}

struct Level<'a>(&'a Cell<usize>);

impl Drop for Level<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

fn enter<E: serde::de::Error>(depth: &Cell<usize>, max: usize) -> Result<Level<'_>, E> {
    let next = depth.get() + 1;
    if next > max {
        return Err(E::custom(format!(
            "wire payload nested deeper than {max} levels"
        )));
    }
    depth.set(next);
    Ok(Level(depth))
}

struct Bounded<'a, D> {
    inner: D,
    depth: &'a Cell<usize>,
    max: usize,
}

struct BoundedVisitor<'a, V> {
    inner: V,
    depth: &'a Cell<usize>,
    max: usize,
}

struct BoundedSeed<'a, S> {
    inner: S,
    depth: &'a Cell<usize>,
    max: usize,
}

struct BoundedAccess<'a, A> {
    inner: A,
    depth: &'a Cell<usize>,
    max: usize,
}

impl<'a, 'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for BoundedSeed<'a, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<S::Value, D::Error> {
        self.inner.deserialize(Bounded {
            inner: deserializer,
            depth: self.depth,
            max: self.max,
        })
    }
}

macro_rules! forward_deserialize {
    ($($method:ident),* $(,)?) => {
        $(
            fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
                self.inner.$method(BoundedVisitor {
                    inner: visitor,
                    depth: self.depth,
                    max: self.max,
                })
            }
        )*
    };
}

impl<'a, 'de, D: Deserializer<'de>> Deserializer<'de> for Bounded<'a, D> {
    type Error = D::Error;

    forward_deserialize!(
        deserialize_any,
        deserialize_bool,
        deserialize_i8,
        deserialize_i16,
        deserialize_i32,
        deserialize_i64,
        deserialize_i128,
        deserialize_u8,
        deserialize_u16,
        deserialize_u32,
        deserialize_u64,
        deserialize_u128,
        deserialize_f32,
        deserialize_f64,
        deserialize_char,
        deserialize_str,
        deserialize_string,
        deserialize_bytes,
        deserialize_byte_buf,
        deserialize_option,
        deserialize_unit,
        deserialize_seq,
        deserialize_map,
        deserialize_identifier,
        deserialize_ignored_any,
    );

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.inner.deserialize_unit_struct(
            name,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.inner.deserialize_newtype_struct(
            name,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.inner.deserialize_tuple(
            len,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.inner.deserialize_tuple_struct(
            name,
            len,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.inner.deserialize_struct(
            name,
            fields,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.inner.deserialize_enum(
            name,
            variants,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}

macro_rules! forward_visit {
    ($($method:ident: $ty:ty),* $(,)?) => {
        $(
            fn $method<E: serde::de::Error>(self, value: $ty) -> Result<V::Value, E> {
                self.inner.$method(value)
            }
        )*
    };
}

impl<'a, 'de, V: Visitor<'de>> Visitor<'de> for BoundedVisitor<'a, V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.inner.expecting(formatter)
    }

    forward_visit!(
        visit_bool: bool,
        visit_i8: i8,
        visit_i16: i16,
        visit_i32: i32,
        visit_i64: i64,
        visit_i128: i128,
        visit_u8: u8,
        visit_u16: u16,
        visit_u32: u32,
        visit_u64: u64,
        visit_u128: u128,
        visit_f32: f32,
        visit_f64: f64,
        visit_char: char,
        visit_str: &str,
        visit_borrowed_str: &'de str,
        visit_string: String,
        visit_bytes: &[u8],
        visit_borrowed_bytes: &'de [u8],
        visit_byte_buf: Vec<u8>,
    );

    fn visit_none<E: serde::de::Error>(self) -> Result<V::Value, E> {
        self.inner.visit_none()
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<V::Value, E> {
        self.inner.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<V::Value, D::Error> {
        let _level = enter::<D::Error>(self.depth, self.max)?;
        self.inner.visit_some(Bounded {
            inner: deserializer,
            depth: self.depth,
            max: self.max,
        })
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<V::Value, D::Error> {
        let _level = enter::<D::Error>(self.depth, self.max)?;
        self.inner.visit_newtype_struct(Bounded {
            inner: deserializer,
            depth: self.depth,
            max: self.max,
        })
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<V::Value, A::Error> {
        let _level = enter::<A::Error>(self.depth, self.max)?;
        self.inner.visit_seq(BoundedAccess {
            inner: seq,
            depth: self.depth,
            max: self.max,
        })
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<V::Value, A::Error> {
        let _level = enter::<A::Error>(self.depth, self.max)?;
        self.inner.visit_map(BoundedAccess {
            inner: map,
            depth: self.depth,
            max: self.max,
        })
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<V::Value, A::Error> {
        let _level = enter::<A::Error>(self.depth, self.max)?;
        self.inner.visit_enum(BoundedAccess {
            inner: data,
            depth: self.depth,
            max: self.max,
        })
    }
}

impl<'a, 'de, A: SeqAccess<'de>> SeqAccess<'de> for BoundedAccess<'a, A> {
    type Error = A::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, A::Error> {
        self.inner.next_element_seed(BoundedSeed {
            inner: seed,
            depth: self.depth,
            max: self.max,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

impl<'a, 'de, A: MapAccess<'de>> MapAccess<'de> for BoundedAccess<'a, A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        self.inner.next_key_seed(BoundedSeed {
            inner: seed,
            depth: self.depth,
            max: self.max,
        })
    }

    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value, A::Error> {
        self.inner.next_value_seed(BoundedSeed {
            inner: seed,
            depth: self.depth,
            max: self.max,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

impl<'a, 'de, A: EnumAccess<'de>> EnumAccess<'de> for BoundedAccess<'a, A> {
    type Error = A::Error;
    type Variant = BoundedAccess<'a, A::Variant>;

    fn variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<(T::Value, Self::Variant), A::Error> {
        let (value, variant) = self.inner.variant_seed(BoundedSeed {
            inner: seed,
            depth: self.depth,
            max: self.max,
        })?;
        Ok((
            value,
            BoundedAccess {
                inner: variant,
                depth: self.depth,
                max: self.max,
            },
        ))
    }
}

impl<'a, 'de, A: VariantAccess<'de>> VariantAccess<'de> for BoundedAccess<'a, A> {
    type Error = A::Error;

    fn unit_variant(self) -> Result<(), A::Error> {
        self.inner.unit_variant()
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, A::Error> {
        self.inner.newtype_variant_seed(BoundedSeed {
            inner: seed,
            depth: self.depth,
            max: self.max,
        })
    }

    fn tuple_variant<V: Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, A::Error> {
        self.inner.tuple_variant(
            len,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, A::Error> {
        self.inner.struct_variant(
            fields,
            BoundedVisitor {
                inner: visitor,
                depth: self.depth,
                max: self.max,
            },
        )
    }
}
