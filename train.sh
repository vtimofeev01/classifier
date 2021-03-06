LABEL=identificable
DATASET=/home/imt/dataset/dataset_for_multilabel_classification
LABEL_PATH="$DATASET"/"$LABEL"
WORK_DIR=/home/imt/work/pyqt5-images-classifier/dataset
echo "label:       $LABEL"
echo "dataset:     $DATASET"
echo "labels file: $LABEL_PATH"
echo "work dir:    $WORK_DIR"

python3 /home/imt/work/pyqt5-images-classifier/prepare_train_label_files_from_dataset_normal_and_numeric.py \
    --attributes_file "$LABEL_PATH"/results.csv \
    --images_dir $DATASET


python3 train.py --images_dir $DATASET \
                 --train_file dataset/partial_big_train.csv \
                 --work_dir $WORK_DIR \
                 --n_epochs 300 \
                 --attributes_file "$WORK_DIR"/data.csv \
                 --val_file "$WORK_DIR"/partial_big_val.csv \
                 --batch_size 16 \
                 --num_workers 10 \
                 --graph_out false \
                 --checkpoint_best "$LABEL"_checkpoint_big


python3 test.py --checkpoint dataset/save/"$LABEL"_checkpoint_big.pth \
                --images_dir $DATASET \
                --test_file "$WORK_DIR"/partial_big_val.csv \
                --attributes_file "$WORK_DIR"/data.csv \
                --workdir "$WORK_DIR"/save \
                --visgrid True \
                --modelname "$LABEL"_classifier


cp "$WORK_DIR"/files_to_check.csv "$LABEL_PATH"/files_to_check.csv