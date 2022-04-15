rlaunch --cpu 8 --gpu 4 --memory 100000 --charged-group v_detection \
--priority Medium --preemptible no \
-- python extract_features.py --mode caffe \
         --num-cpus 8 --gpu '0,1,2,3' \
         --extract-mode bboxes \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \
         --image-dir datasets/gqa/images/ --out-dir datasets/gqa/gqa_output_features --resume