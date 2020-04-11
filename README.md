# classifier

I needed fast utility to make an csv-file (image-name,label) for clissification in deep learning

I need to work with continually growing dataset. Standard utilities are TOO clever and not fast

Here i can do up to the 3-5 images per second

CLI attributes: 

    --img-dir - a dataset folder with pictures 

    --l - name of attribut (label). the system creates for each label
    an subfolder in dataset-folder containing result.csv
    if you starts more times this results.csv will be loaded automatically
                
    --lv - list of values if --l coma separated without spaces between
    if you starts many times the same --l system will load and add label values 
    already used im results.csv
                
                
Interface is easy

0..9 to select values 
a - d navigate thought images
space - go to the next unlabeled image

If anyone has any ideas - i can add it

