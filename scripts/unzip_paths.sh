unzip preproc/training_paths1.zip -d preproc
unzip preproc/training_paths2.zip -d preproc

unzip preproc/test.paths.zip  preproc/test.paths

cat preproc/xa* > preproc/training.paths
unzip preproc/validation.paths.zip preproc/validation.paths
