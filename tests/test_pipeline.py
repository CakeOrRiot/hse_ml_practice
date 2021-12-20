import os

from scripts.pipeline import Pipeline


def test_result():
    path = os.path.abspath('data')
    pipeline = Pipeline(path)
    pipeline.run()
    file = f"{path}/submission.csv"
    created = os.path.exists(file)
    assert created
    os.remove(file)


test_result()
