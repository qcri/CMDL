from gensim.models.fasttext import load_facebook_vectors

ft_model_path = '../resources/fasttext/cc.en.300.bin'
ft_op_path = '../resources/fasttext/cc/cc.en.300.bin'

model = load_facebook_vectors(ft_model_path)
print('loaded model')
model.init_sims(replace=True)
model.save(ft_op_path)
