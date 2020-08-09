#python3 test.py --checkpoint dataset/save/checkpoint-best.pth \
#                --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
#                --test_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
#                --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
#                --workdir dataset/save \
#                --visgrid True

#python3 test.py --checkpoint dataset/save/checkpoint_small.pth \
#                --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
#                --test_file /home/imt/work/pyqt5-images-classifier/dataset/partial_small_val.csv \
#                --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
#                --workdir dataset/save
#
python3 test.py --checkpoint dataset/save/checkpoint_big.pth \
                --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
                --test_file /home/imt/work/pyqt5-images-classifier/dataset/partial_big_val.csv \
                --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
                --workdir dataset/save \
                --visgrid True \
                --modelname persons_classifier
