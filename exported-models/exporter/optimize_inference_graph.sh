#!/bin/bash

tensorflow_path=../../../../TensorFlow/tensorflow

$tensorflow_path/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=../converted/frozen_inference_graph.pb \
--out_graph=../converted/optimzed_inference_graph.pb \
--inputs='image_tensor' \
--outputs='detection_boxes,detection_scores,detection_classes,num_detections' \
--transforms='
strip_unused_nodes()
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms'
