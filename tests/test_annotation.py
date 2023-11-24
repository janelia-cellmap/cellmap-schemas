from typing import Literal
from cellmap_schemas.annotation import (
	AnnotationArray,
	AnnotationArrayAttrs,
	AnnotationGroup,
	AnnotationGroupAttrs,
	CropGroup,
	CropGroupAttrs,
	SemanticSegmentation,
	wrap_attributes,
)
import numpy as np
import zarr


def test_cropgroup():
	ClassNamesT = Literal['foo', 'bar']
	class_names = ['foo', 'bar']
	ann_type = SemanticSegmentation(encoding={'absent': 0})
	arrays = [np.zeros(10) for class_name in class_names]
	crop_group_attrs = CropGroupAttrs[ClassNamesT](
		name='foo',
		description='a description',
		created_by=['person_a', 'person_b'],
		created_with=['amira'],
		start_date='2022-03-01',
		end_date='2022-04-01',
		duration_days=4,
		protocol_uri=None,
		class_names=class_names,
	)

	ann_groups = {}

	for array, class_name in zip(arrays, class_names):
		ann_array_attrs = AnnotationArrayAttrs[ClassNamesT](
			class_name=class_name, complement_counts={'absent': 0}, annotation_type=ann_type
		)
		ann_array = AnnotationArray.from_array(array, attrs=wrap_attributes(ann_array_attrs))

		ann_group_attrs = AnnotationGroupAttrs[ClassNamesT](
			class_name=class_name, annotation_type=ann_type
		)

		# check that an extra attribute is OK
		ann_group = AnnotationGroup(
			members={'s0': ann_array},
			attrs={**wrap_attributes(ann_group_attrs).dict(), **{'foo': 10}},
		)
		ann_groups[class_name] = ann_group

	crop_group = CropGroup(members=ann_groups, attrs=wrap_attributes(crop_group_attrs))

	stored = crop_group.to_zarr(zarr.MemoryStore(), path='test')
	observed = CropGroup.from_zarr(stored)
	assert observed == crop_group

	## todo: insert jsonschema validation step
