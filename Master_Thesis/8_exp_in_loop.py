#%%
# import algorithms
from LearningPipeline import OneDCNN, OnlinePrototypicalNetwork, OfflinePrototypicalNetwork
from LearningPipeline import TransferLearning, TransferPrototypicalNetwork
# import query strategy
from LearningPipeline import random_sampling, random_batch_sampling
from modAL.uncertainty import uncertainty_sampling
from modAL.batch import uncertainty_batch_sampling
# import evaluator
from LearningPipeline import Evaluator
# import data generator
from DataGeneration import GenerateHARData, GenerateHAPTData

#%%
query_strategies = [random_sampling, uncertainty_sampling]
estimators = [OneDCNN(), OnlinePrototypicalNetwork(), TransferPrototypicalNetwork()]
for estimator in estimators:
	for query_strategy in query_strategies:
		evaluator = Evaluator(
			data_generator = GenerateHARData(), 
			estimator = estimator, 
			query_strategy = query_strategy,
			init_size=1
		)
		evaluator.run(n_queries=1000, iteration=5, visual=False, save=True)
		estimator_name = f"{estimator}".split(" ")[0].split(".")[1]
		query_strategy_name = f"{query_strategy}".split(" ")[1]
		print(f"current: {estimator_name}_{query_strategy_name}")

#%%