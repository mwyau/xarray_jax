# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax
import xarray_jax.pytree


class PytreeTest(absltest.TestCase):

  def test_flatten_unflatten_variable(self):
    attrs = {
        'standard_name': 'air_temperature',
        'units': 'K',
        'valid_range': np.array([180.0, 330.0]),
    }
    variable = xarray.Variable(
        ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32), attrs=attrs)
    children, aux = xarray_jax.pytree._flatten_variable(variable)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_variable(aux, children)
    self.assertTrue(variable.identical(roundtrip))

  def test_hashable_attrs_array_aware_equality(self):
    attrs = {
        'units': 'K',
        'valid_range': np.array([180.0, 330.0]),
        'flags': [1, 2, 4],
    }
    hashable_attrs = xarray_jax.pytree._HashableAttrs(attrs)
    equal_attrs = xarray_jax.pytree._HashableAttrs({
        'units': 'K',
        'valid_range': np.array([180.0, 330.0]),
        'flags': [1, 2, 4],
    })
    different_attrs = xarray_jax.pytree._HashableAttrs({
        'units': 'degC',
        'valid_range': np.array([180.0, 330.0]),
        'flags': [1, 2, 4],
    })

    self.assertEqual(hash(hashable_attrs), hash(equal_attrs))
    self.assertEqual(hashable_attrs, equal_attrs)
    self.assertNotEqual(hashable_attrs, different_attrs)

    attrs['units'] = 'degC'
    self.assertEqual(hashable_attrs['units'], 'K')

  def test_hashable_attrs_nested_mapping_equality(self):
    attrs1 = {
        'metadata': {
            'units': 'K',
            'valid_range': np.array([180.0, 330.0]),
        }
    }
    attrs2 = {
        'metadata': {
            'units': 'K',
            'valid_range': np.array([180.0, 330.0]),
        }
    }
    attrs3 = {
        'metadata': {
            'units': 'degC',
            'valid_range': np.array([180.0, 330.0]),
        }
    }
    attrs4 = {
        'metadata': {
            'units': 'K',
            'valid_range': np.array([180.0, 331.0]),
        }
    }

    wrapped1 = xarray_jax.pytree._HashableAttrs(attrs1)
    wrapped2 = xarray_jax.pytree._HashableAttrs(attrs2)
    wrapped3 = xarray_jax.pytree._HashableAttrs(attrs3)
    wrapped4 = xarray_jax.pytree._HashableAttrs(attrs4)

    self.assertEqual(wrapped1, wrapped2)
    self.assertNotEqual(wrapped1, wrapped3)
    self.assertNotEqual(wrapped1, wrapped4)
    self.assertEqual(hash(wrapped1), hash(wrapped2))
    hash(wrapped1)
    hash(wrapped2)

  def test_flatten_unflatten_data_array(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        name='temperature',
        attrs={'standard_name': 'air_temperature', 'units': 'K'},
        coords={
            'lat': xarray.Variable(
                ('lat',), np.arange(3), attrs={'units': 'degrees_north'})},
        jax_coords={
            'lon': xarray.Variable(
                ('lon',), jnp.arange(4) * 10,
                attrs={'units': 'degrees_east'})},
    )
    children, aux = xarray_jax.pytree._flatten_data_array(data_array)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_data_array(aux, children)
    self.assertTrue(data_array.identical(roundtrip))
    xarray.testing.assert_identical(
        jax.device_get(data_array), jax.device_get(roundtrip))

  def test_flatten_unflatten_dataset(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        data_vars={
            'foo': xarray.Variable(
                ('lat', 'lon'), foo, attrs={'units': 'K'}),
            'bar': xarray.Variable(
                ('time', 'lat', 'lon'), bar,
                attrs={'standard_name': 'air_temperature'})},
        coords={
            'time': xarray.Variable(
                ('time',), np.arange(2), attrs={'axis': 'T'}),
            'lat': xarray.Variable(
                ('lat',), np.arange(3) * 10,
                attrs={'units': 'degrees_north'})},
        attrs={
            'title': 'Weather',
            'valid_range': np.array([180.0, 330.0]),
            'flags': [1, 2, 4],
        },
        jax_coords={
            'lon': xarray.Variable(
                ('lon',), jnp.arange(4) * 10,
                attrs={'units': 'degrees_east'})})
    children, aux = xarray_jax.pytree._flatten_dataset(dataset)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_dataset(aux, children)
    self.assertTrue(dataset.identical(roundtrip))
    xarray.testing.assert_identical(
        jax.device_get(dataset), jax.device_get(roundtrip))

  def test_attrs_are_part_of_jit_identity(self):
    kelvin = xarray_jax.DataArray(
        data=jnp.ones((2,)), dims=('x',), attrs={'units': 'K'})
    celsius = xarray_jax.DataArray(
        data=jnp.ones((2,)), dims=('x',), attrs={'units': 'degC'})

    @jax.jit
    def scale(value):
      factor = 2 if value.attrs['units'] == 'K' else 3
      return value.data * factor

    np.testing.assert_array_equal(jax.device_get(scale(kelvin)), [2, 2])
    np.testing.assert_array_equal(jax.device_get(scale(celsius)), [3, 3])

  def test_dataset_attrs_are_part_of_jit_identity(self):
    kelvin = xarray_jax.Dataset(
        {'temperature': (('x',), jnp.ones((2,)))}, attrs={'units': 'K'})
    celsius = xarray_jax.Dataset(
        {'temperature': (('x',), jnp.ones((2,)))}, attrs={'units': 'degC'})

    @jax.jit
    def scale(value):
      factor = 2 if value.attrs['units'] == 'K' else 3
      return value['temperature'].data * factor

    np.testing.assert_array_equal(jax.device_get(scale(kelvin)), [2, 2])
    np.testing.assert_array_equal(jax.device_get(scale(celsius)), [3, 3])

  def test_static_coordinate_attrs_are_part_of_pytree_identity(self):
    degrees = xarray_jax.DataArray(
        data=jnp.ones((2,)),
        dims=('lat',),
        coords={'lat': xarray.Variable(
            ('lat',), np.arange(2), attrs={'units': 'degrees_north'})})
    radians = xarray_jax.DataArray(
        data=jnp.ones((2,)),
        dims=('lat',),
        coords={'lat': xarray.Variable(
            ('lat',), np.arange(2), attrs={'units': 'radians'})})

    _, degrees_aux = xarray_jax.pytree._flatten_data_array(degrees)
    _, radians_aux = xarray_jax.pytree._flatten_data_array(radians)
    self.assertNotEqual(degrees_aux, radians_aux)
    self.assertNotEqual(degrees_aux[1], radians_aux[1])

    @jax.jit
    def scale(value):
      factor = (2 if value.coords['lat'].attrs['units'] == 'degrees_north'
                else 3)
      return value.data * factor

    np.testing.assert_array_equal(jax.device_get(scale(degrees)), [2, 2])
    np.testing.assert_array_equal(jax.device_get(scale(radians)), [3, 3])

  def test_flatten_unflatten_datatree(self):
    # Coords to be inherited from the parent dataset, we include one jax
    # coord and one not to check both code paths
    parent_dataset = xarray_jax.Dataset(
        jax_coords={'time': xarray.Variable(('time',), np.arange(2))},
        coords={'lon': xarray.Variable(
            ('lon',), np.arange(4) * 10,
            attrs={'units': 'degrees_east'})},
        attrs={'node': 'parent'})

    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    child_dataset = xarray_jax.Dataset(
        {'bar': xarray.Variable(
            ('time', 'lat', 'lon'), bar, attrs={'units': 'K'})},
        coords={'lat': xarray.Variable(
            ('lat',), np.arange(3), attrs={'units': 'degrees_north'})},
        attrs={'node': 'child'})

    datatree = xarray.DataTree(
        dataset=parent_dataset,
        children={'child': xarray.DataTree(dataset=child_dataset)})

    children, aux = xarray_jax.pytree._flatten_datatree(datatree)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax.pytree._unflatten_datatree(aux, children)
    self.assertTrue(datatree.identical(roundtrip))

  def test_flatten_unflatten_added_dim(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    leaves, treedef = jax.tree_util.tree_flatten(data_array)
    leaves = [jnp.expand_dims(x, 0) for x in leaves]
    with xarray_jax.dims_change_on_unflatten(lambda dims: ('new',) + dims):
      with_new_dim = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(('new', 'lat', 'lon'), with_new_dim.dims)
    xarray.testing.assert_identical(
        jax.device_get(data_array),
        jax.device_get(with_new_dim.isel(new=0)))

  def test_map_added_dim(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    with xarray_jax.dims_change_on_unflatten(lambda dims: ('new',) + dims):
      with_new_dim = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0),
                                            data_array)
    self.assertEqual(('new', 'lat', 'lon'), with_new_dim.dims)
    xarray.testing.assert_identical(
        jax.device_get(data_array),
        jax.device_get(with_new_dim.isel(new=0)))

  def test_map_remove_dim(self):
    foo = jnp.ones((1, 3, 4), dtype=np.float32)
    bar = jnp.ones((1, 2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        data_vars={'foo': (('batch', 'lat', 'lon'), foo),
                   'bar': (('batch', 'time', 'lat', 'lon'), bar)},
        coords={
            'batch': np.array([123]),
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
            'lon': np.arange(4) * 10})
    with xarray_jax.dims_change_on_unflatten(lambda dims: dims[1:]):
      with_removed_dim = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 0),
                                                dataset)
    self.assertEqual(('lat', 'lon'), with_removed_dim['foo'].dims)
    self.assertEqual(('time', 'lat', 'lon'), with_removed_dim['bar'].dims)
    self.assertNotIn('batch', with_removed_dim.dims)
    self.assertNotIn('batch', with_removed_dim.coords)
    xarray.testing.assert_identical(
        jax.device_get(dataset.isel(batch=0, drop=True)),
        jax.device_get(with_removed_dim))


class NonArrayLeafWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (type(leaf).__name__, leaf) for leaf in [
          True,
          None,
          42,
          3.14,
          object(),
          jax.stages.ArgInfo(
              jax.core.ShapedArray(shape=(2, 3), dtype=jnp.float32),
              donated=True
          ),
          jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
      ]
  )
  def test_preserves_leaf_identity(self, leaf):
    var = xarray.Variable(('x', 'y'), jnp.ones((2, 3)))
    _, aux = xarray_jax.pytree._flatten_variable(var)
    unflattened_var = xarray_jax.pytree._unflatten_variable(aux, (leaf,))

    # Preserves leaf identity
    self.assertIsInstance(
        unflattened_var.data, xarray_jax.NonArrayLeafWrapper
    )
    self.assertIs(unflattened_var.data.leaf, leaf)

  @parameterized.named_parameters(
      ('int', 42, (0, 0), np.int32),
      ('bool', True, (0, 0), np.bool_),
      ('float', 3.14, (0, 0), np.float32),
      ('ShapeDtypeStruct', jax.ShapeDtypeStruct(
          shape=(2, 3), dtype=jnp.float32), (2, 3), np.float32),
      ('None', None, (0, 0), np.dtype(object)),
      ('object', object(), (0, 0), np.dtype(object)),
  )
  def test_creates_valid_xarray_objects(
      self, leaf, expected_shape, expected_dtype
  ):
    var = xarray.Variable(('x', 'y'), jnp.ones((2, 3)))
    _, aux = xarray_jax.pytree._flatten_variable(var)
    unflattened_var = xarray_jax.pytree._unflatten_variable(
        aux, (leaf,)
    )

    # Returns a valid xarray object
    self.assertEqual(unflattened_var.shape, expected_shape)
    self.assertEqual(unflattened_var.dims, var.dims)
    self.assertEqual(unflattened_var.dtype, expected_dtype)

    # Allows basic operations
    self.assertEqual(unflattened_var.data.ndim, var.ndim)
    self.assertEqual(unflattened_var.data.size, np.prod(expected_shape))

  def test_creates_valid_xarray_objects_scalar(self):
    # Ensure shape=() when dims=() and leaf has no shape
    var, leaf = xarray.Variable((), jnp.array(1.0)), 7
    _, aux = xarray_jax.pytree._flatten_variable(var)
    unflattened_var = xarray_jax.pytree._unflatten_variable(aux, (leaf,))
    self.assertEqual(unflattened_var.shape, ())
    self.assertEqual(unflattened_var.dims, var.dims)
    self.assertEqual(unflattened_var.dtype, jnp.int32)
    self.assertEqual(unflattened_var.data.ndim, 0)
    self.assertEqual(unflattened_var.data.size, np.prod(()))

  def test_unsupported_operations_raise_error(self):
    var, leaf = xarray.Variable(('x',), jnp.ones((2,))), 42
    _, aux = xarray_jax.pytree._flatten_variable(var)
    unflattened_var = xarray_jax.pytree._unflatten_variable(
        aux, (leaf,)
    )
    wrapped_data = unflattened_var.data

    with self.assertRaisesRegex(TypeError, 'NumPy ufunc'):
      np.add(wrapped_data, 1)
    with self.assertRaisesRegex(TypeError, 'NumPy function'):
      np.sum(wrapped_data)
    with self.assertRaisesRegex(TypeError, 'Indexing is not supported'):
      _ = wrapped_data[0]

  def test_roundtrip_with_non_array_leaves(self):
    original_dataset = xarray_jax.Dataset(
        data_vars={
            'foo': (('x', 'y'), jnp.ones((2, 3))),
            'bar': (('y', 'z'), jnp.zeros((3, 4))),
        },
        coords={'x': np.arange(2)},
        jax_coords={'y': jnp.arange(3) * 10},
    )

    # Map to ShapeDtypeStruct.
    struct_dataset = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        original_dataset)

    self.assertIsInstance(
        struct_dataset['foo'].data, xarray_jax.NonArrayLeafWrapper)
    self.assertIsInstance(
        struct_dataset['bar'].data, xarray_jax.NonArrayLeafWrapper)
    self.assertIsInstance(
        struct_dataset['y'].data, xarray_jax.NonArrayLeafWrapper)

    self.assertIsInstance(
        struct_dataset['foo'].data.leaf, jax.ShapeDtypeStruct)
    self.assertIsInstance(
        struct_dataset['bar'].data.leaf, jax.ShapeDtypeStruct)
    self.assertIsInstance(
        struct_dataset['y'].data.leaf, jax.ShapeDtypeStruct)

    # Map back to arrays.
    reconstructed_dataset = jax.tree_util.tree_map(
        lambda x: jnp.ones(x.shape, x.dtype),
        struct_dataset)

    self.assertIsInstance(reconstructed_dataset['foo'].data, jax.Array)
    self.assertIsInstance(reconstructed_dataset['bar'].data, jax.Array)
    self.assertIsInstance(reconstructed_dataset['y'].data, jax.Array)

    expected_dataset = xarray_jax.Dataset(
        data_vars={
            'foo': (('x', 'y'), jnp.ones((2, 3))),
            'bar': (('y', 'z'), jnp.ones((3, 4))),
        },
        coords={'x': np.arange(2)},
        jax_coords={'y': jnp.ones(original_dataset.y.shape,
                                  dtype=original_dataset.y.dtype)},
    )
    xarray.testing.assert_identical(
        jax.device_get(reconstructed_dataset),
        jax.device_get(expected_dataset),
    )

  def test_two_arg_tree_map_roundtrip_with_non_array_leaves(self):
    original_dataset = xarray_jax.Dataset(
        data_vars={
            'foo': (('x', 'y'), jnp.ones((2, 3), dtype=jnp.float32)),
            'bar': (('y', 'z'), jnp.zeros((3, 4), dtype=jnp.float32)),
        },
        jax_coords={'y': jnp.arange(3) * 10},
    )

    # Map to arbitrary non-array leaves (booleans).
    bool_dataset = jax.tree_util.tree_map(lambda _: True, original_dataset)

    # Use tree_map to reconstruct arrays using the original shapes/dtypes.
    reconstructed = jax.tree_util.tree_map(
        lambda orig, _tag: jnp.ones(orig.shape, orig.dtype),
        original_dataset,
        bool_dataset,
    )

    expected_dataset = xarray_jax.Dataset(
        data_vars={
            'foo': (('x', 'y'), jnp.ones((2, 3), dtype=jnp.float32)),
            'bar': (('y', 'z'), jnp.ones((3, 4), dtype=jnp.float32)),
        },
        jax_coords={'y': jnp.ones(original_dataset.y.shape,
                                  dtype=original_dataset.y.dtype)},
    )
    xarray.testing.assert_identical(
        jax.device_get(reconstructed),
        jax.device_get(expected_dataset),
    )

if __name__ == '__main__':
  absltest.main()
