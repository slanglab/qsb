unzip preproc/training_paths1.zip -d preproc
unzip preproc/training_paths2.zip -d preproc
unzip preproc/test_paths.zip -d preproc

cat preproc/xa* > preproc/training.paths
