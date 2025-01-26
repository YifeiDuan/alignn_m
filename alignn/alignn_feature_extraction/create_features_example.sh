for filename in matbench_jdft2d_exfoliation_en_fold_0/mb-jdft2d-*; do
    python ../pretrained_activation_mb.py --prop_name ../matbench_jdft2d_exfoliation_en --fold 0 --file_format poscar --file_path "$filename" --output_path ../embed_matbench_jdft2d/fold_0
done
