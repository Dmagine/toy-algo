import anylearn


anylearn.init_sdk('http://10.101.67.7:30080', 'yhuang', 'Anylearn2021!')

task, _, _, _ = anylearn.quick_train(
    algorithm_name="ToyAlgo",
    algorithm_dir=".",
    algorithm_force_update=True,
    entrypoint="python train.py",
    output="results",
    hyperparams={
        'epochs': 12,
        'model': 'BiT-M-R50x1',
    },
)
print(task)
