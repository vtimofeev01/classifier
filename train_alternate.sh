#python3 /home/imt/work/pyqt5-images-classifier/prepare_train_label_files_from_dataset_normal_and_numeric.py


python3 train_alternate.py --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
                 --train_file dataset/partial_big_train.csv \
                 --work_dir /home/imt/work/pyqt5-images-classifier/dataset \
                 --n_epochs 250 \
                 --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
                 --val_file /home/imt/work/pyqt5-images-classifier/dataset/partial_big_val.csv \
                 --batch_size 16 \
                 --num_workers 10 \
                 --checkpoint_best checkpoint_big


#python3 test.py --checkpoint dataset/save/checkpoint_big.pth \
#                --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
#                --test_file /home/imt/work/pyqt5-images-classifier/dataset/partial_big_val.csv \
#                --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
#                --workdir dataset/save \
#                --visgrid True \
#                --modelname persons_classifier