import data_gen, pipeline, model
data_gen.make(); pipeline.run(); model.train_initial(); model.evolve()
print("\nDone. Now run:  streamlit run app.py")
