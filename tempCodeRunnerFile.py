dataset.prepare_data()
dataset.setup(stage="test")
trainer = Trainer()
y_hat = trainer.predict(model, datamodule=dataset)
