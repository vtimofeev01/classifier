python3 train.py --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
                 --train_file dataset/train.csv \
                 --work_dir /home/imt/work/pyqt5-images-classifier/dataset \
                 --n_epochs 500 \
                 --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
                 --val_file /home/imt/work/pyqt5-images-classifier/dataset/val.csv \
                 --batch_size 16 \
                 --num_workers 10 \
                 --graph_out false

python3 train.py --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
                 --train_file dataset/partial_small_train.csv \
                 --work_dir /home/imt/work/pyqt5-images-classifier/dataset \
                 --n_epochs 500 \
                 --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
                 --val_file dataset/partial_small_val.csv \
                 --batch_size 16 \
                 --num_workers 10 \
                 --graph_out false \
                 --checkpoint_best checkpoint_small

python3 train.py --images_dir /home/imt/dataset/dataset_for_multilabel_classification \
                 --train_file dataset/partial_big_train.csv \
                 --val_file dataset/partial_big_val.csv \
                 --work_dir /home/imt/work/pyqt5-images-classifier/dataset \
                 --n_epochs 500 \
                 --attributes_file /home/imt/work/pyqt5-images-classifier/dataset/data.csv \
                 --batch_size 16 \
                 --num_workers 10 \
                 --graph_out false \
                 --checkpoint_best checkpoint_big