import torch
from torch.nn import functional as F
from pdb import set_trace as bp
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import pickle as pkl
import numpy as np
import json

# Local modules
# import models_pairwise
import lightning_rnn as models
import util

def lambda_handler(event, context):
	query = event['chatbotMessage']
	response = simpleFilterResponse(query)
	response = json.dumps(response)
	return {'statusCode': 200, 'body': response, 'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}}

def run_trainer(parameters):
	model = models.Classifier(parameters)
	print("Initializing the model")

	pl.seed_everything(parameters['train_seed'], workers=True)
	
	loggerName = parameters['logger_name'] + "_" + str(parameters['train_seed'])
	wandb_logger = WandbLogger(project='vpCodeRemake', name=loggerName)
	wandb_logger.experiment.config.update(parameters)

	print("Setting up the trainer")
	trainer = pl.Trainer(
	accelerator="auto",
	devices=[parameters['gpu_num']] if torch.cuda.is_available() else None,  # limiting got iPython runs
	max_epochs=parameters['max_epochs'],
	callbacks=[TQDMProgressBar(refresh_rate=20), 
		EarlyStopping(monitor="val_f1", mode="max", patience=parameters['patience']), 
		ModelCheckpoint(monitor='val_f1', dirpath='modelStore/', filename=loggerName + '-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='max'),
		# pl.callbacks.StochasticWeightAveraging(swa_epoch_start=70, swa_lrs=0.005),
		],
	logger=wandb_logger,
	deterministic=True
	)
	print("Starting the training")
	trainer.fit(model)
	with open(loggerName + '_metricsDict.pkl', 'wb') as f:
		pkl.dump(model.metricsDict, f)
	trainer.test(model, ckpt_path='best')


def load_model(parameters, model_path=None, cpu=False):
	model = models.Classifier(parameters)
	if cpu:
		if model_path is None:
			model.load_state_dict(torch.load(parameters['model_path'], map_location=torch.device('cpu'))['state_dict'])
		else:
			newDict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
			del newDict['model.embed.embd.position_ids']
			# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
			model.load_state_dict(newDict)
	elif model_path is None:
		model.load_state_dict(torch.load(parameters['model_path'])['state_dict'])
	else:
		model.load_state_dict(torch.load(model_path)['state_dict'])
	return model

def evaluate_model(model, parameters):
	# Evaluate the model on the test set
	trainer = pl.Trainer(
	accelerator="auto",
	devices=[parameters['gpu_num']] if torch.cuda.is_available() else None,  # limiting got iPython runs
	deterministic=True
	)
	outDictList = []

	model.knn = False
	trainer.test(model)
	metrics = model.metricsDict
	metrics['evalParam'] = "no_knn"
	metrics['model_path'] = parameters['model_path']
	outDictList.append(metrics)

	model.knn = True
	trainer.test(model)
	metrics = model.metricsDict
	metrics['evalParam'] = "knn"
	metrics['model_path'] = parameters['model_path']
	outDictList.append(metrics)

	df = pd.DataFrame(outDictList)
	# tsvName = parameters['logger_name'] + "_" + "_eval.tsv"
	tsvName = "cosi_knn_eval.tsv"
	df.to_csv(tsvName, sep="\t", index=False, mode='a')

def run_inference(model, parameters):
	trainer = pl.Trainer(
	accelerator="auto",
	devices=[parameters['gpu_num']] if torch.cuda.is_available() else None,  # limiting got iPython runs
	deterministic=True
	)
	model.eval()
	model.model.eval()
	# model.on_test_start()
	model.knn = False
	utterance = ""

	labelToResponse = labelResponseMap()

	while utterance != "quit" and utterance != "q":
		utterance = input("Enter an utterance: ")
		outClass = model.infer(utterance)
		print(outClass)
		print(labelToResponse[str(outClass[1])])
	return outClass

def run_batchInference(model, parameters, inFile, outFile="evalTestData_out.tsv"):
	trainer = pl.Trainer(
	accelerator="auto",
	devices=[parameters['gpu_num']] if torch.cuda.is_available() else None,  # limiting got iPython runs
	deterministic=True
	)
	model.eval()
	model.model.eval()
	model.on_test_start()

	# Change this to the label to response mapping for the dataset
	labelToResponse = labelResponseMap()

	outRows = []
	df = pd.read_csv(inFile, sep="\t")
	for i, row in df.iterrows():
		utterance = row['Utterance']
		rowDict = row.to_dict()
		try:
			if len(utterance) > 0:
				trueLabel = row['True_Label']
				outPred= model.infer(utterance)
				bestPred = outPred[1]
				rowDict['Predicted_Label'] = bestPred
				sortedPred = np.argsort(-outPred[-1])
				trueLabelRank = np.where(sortedPred == trueLabel)[0][0]
				predTrueLogitDiff = outPred[-1][bestPred] - outPred[-1][trueLabel]
				rowDict['True_Label_Rank'] = trueLabelRank
				rowDict['Pred_True_Logit_Diff'] = predTrueLogitDiff
		except:
			print("Error on row: ", i)
		outRows.append(rowDict)
	
	outDf = pd.DataFrame(outRows)
	outDf.to_csv(outFile, sep="\t", index=False)
		
	return outPred

def filterInference(modelsDict, parameters):
	trainer = pl.Trainer(
	accelerator="auto",
	devices=[parameters['gpu_num']] if torch.cuda.is_available() else None,
	deterministic=True
	)

	responseDict = {
		"age": {0: "default", 1: "infants: 0 months - 1 year", 2: "toddlers: 1 - 2 years", 3: "preschoolers: 3 - 4 years", 4: "school-aged children: 5 - 12 years", 5: "children: 2 - 12 years", 6: "teens: 13 - 19 years", 7: "young adults: 20 - 30 years", 8: "adults: 31 - 54 years", 9: "seniors: 55 years +"},
		"disability": {0: "default", 1: "all disabilities", 2: "limited mobility", 3: "physical disability", 4: "visual impairment"},
		"health": {0: "default", 1: "chonic illness", 2: "diabetes", 3: "genetic disorder", 4: "pregnant", 5: "cancer"},
		"housing": {0: "default", 1: "homeless", 2: "near homeless", 3: "runaways"},
		"income": {0: "default", 1: "benefit recipients", 2: "low-income"},
		"role": {0: "default", 1: "dependents", 2: "spouses", 3: "caregiver", 4: "parents"}
		}

	for filterName, filterModel in modelsDict.items():
		filterModel.eval()
		filterModel.model.eval()
		filterModel.knn = False
		filterModel.label_dict = responseDict[filterName]
	
	utterance = ""
	labelToResponse = labelResponseMap()


	print("Please tell me more about yourself so we can find the best resource for you. Important details are especially if you have considerations such as age, disability, health, housing, income, or role.")
	while utterance != "quit" and utterance != "q":
		utterance = input("Enter an utterance: ")
		print("We have identified the following considerations for you.")
		for filterName, filterModel in modelsDict.items():
			retVal = filterModel.infer(utterance)
			print("Category: ", filterName)
			for label in retVal:
				print(label)
			print()

		print("Are these correct? If not, please let me know which categories are incorrect. Simply write them in separated by commas.")
		utterance = input("Enter an utterance: ")
		utterance = utterance.split(",")
		for filterName, filterModel in modelsDict.items():
			if filterName in utterance:
				print("Please tell me more about your ", filterName, ".")
				utterance = input("Enter an utterance: ")
				retVal = filterModel.infer(utterance)
				print("Category: ", filterName)
				for label in retVal:
					print(label)
				print()
	return retVal


def labelResponseMap(inFile=None):
	labelToResponse = {
		'0': "Let's find a community garden near you",
		'1': "Let's find food delivery near you.",
		'2': "Let's find a food pantry near you.",
		'3': "Let's find food payment assistance near you.",
		'4': "Let's find meals near you.",
		'5': "Let's find locations that can help with nutrition education near you."
	}

	# TODO: Get mapping from file

	return labelToResponse

def simpleFilterResponse(utterance):
	parameters = util.getParameters()

	parameters['infer'] = True
	parameters['multi_label'] = True
	parameters['label_data'] = "data/daphne/filterData/filterData_labels_filterType_return.json"
	parameters['train_data'] = "data/daphne/filterData/filterData_train_filterType.json"
	parameters['test_data'] = "data/daphne/filterData/filterData_test_filterType.json"
	parameters['val_data'] = "data/daphne/filterData/filterData_val_filterType.json"
	parameters['para_task'] = "None"

	modelPathsDict = {
			"age": "modelStore/filterRun_age_1111-epoch=09-val_loss=0.00.ckpt",
			"disability": "modelStore/filterRun_disability_1111-epoch=09-val_loss=0.00.ckpt",
			"health": "modelStore/filterRun_health_1111-epoch=09-val_loss=0.00.ckpt",
			"housing": "modelStore/filterRun_housing_1111-epoch=09-val_loss=0.00.ckpt",
			"role": "modelStore/filterRun_role_1111-epoch=09-val_loss=0.00.ckpt"
			}
	modelsDict = {}
	for filterName, modelPath in modelPathsDict.items():
		try:
			filterParams = parameters.copy()
			# replace "fiterType" with filterName for each value in params
			for key, value in filterParams.items():
				if type(value) == str:
					value = value.replace("filterType", filterName)
					filterParams[key] = value

			modelsDict[filterName] = load_model(filterParams, model_path=modelPath, cpu=True)
		except:
			continue

	responseDict = {
		"age": {0: "default", 1: "infants: 0 months - 1 year", 2: "toddlers: 1 - 2 years", 3: "preschoolers: 3 - 4 years", 4: "school-aged children: 5 - 12 years", 5: "children: 2 - 12 years", 6: "teens: 13 - 19 years", 7: "young adults: 20 - 30 years", 8: "adults: 31 - 54 years", 9: "seniors: 55 years +"},
		"disability": {0: "default", 1: "all disabilities", 2: "limited mobility", 3: "physical disability", 4: "visual impairment"},
		"health": {0: "default", 1: "chonic illness", 2: "diabetes", 3: "genetic disorder", 4: "pregnant", 5: "cancer"},
		"housing": {0: "default", 1: "homeless", 2: "near homeless", 3: "runaways"},
		"income": {0: "default", 1: "benefit recipients", 2: "low-income"},
		"role": {0: "default", 1: "dependents", 2: "spouses", 3: "caregiver", 4: "parents"}
		}

	outVals = dict()
	for filterName, filterModel in modelsDict.items():
		filterModel.eval()
		filterModel.model.eval()
		filterModel.knn = False
		filterModel.label_dict = responseDict[filterName]
		retVal = filterModel.infer(utterance, return_labelNum=True)
		outVals[filterName] = (retVal[0], retVal[1].tolist(), retVal[2].detach().tolist())
	
	return outVals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
	simpleFilterResponse("Hello world")
	# parameters = util.getParameters()


	# if parameters['eval']:
	# 	model = load_model(parameters)
	# 	evaluate_model(model, parameters)
	# elif parameters['infer']:
	# 	if parameters['multi_label']:
	# 		modelPathsDict = {
	# 		"age": "modelStore/filterRun_age_1111-epoch=09-val_loss=0.00.ckpt",
	# 		"disability": "modelStore/filterRun_disability_1111-epoch=09-val_loss=0.00.ckpt",
	# 		"health": "modelStore/filterRun_health_1111-epoch=09-val_loss=0.00.ckpt",
	# 		"housing": "modelStore/filterRun_housing_1111-epoch=09-val_loss=0.00.ckpt",
	# 		"role": "modelStore/filterRun_role_1111-epoch=09-val_loss=0.00.ckpt"
	# 		}
	# 		modelsDict = {}
	# 		for filterName, modelPath in modelPathsDict.items():
	# 			filterParams = parameters.copy()
	# 			# replace "fiterType" with filterName for each value in params
	# 			for key, value in filterParams.items():
	# 				if type(value) == str:
	# 					value = value.replace("filterType", filterName)
	# 					filterParams[key] = value

	# 			modelsDict[filterName] = load_model(filterParams, model_path=modelPath)
	# 		filterInference(modelsDict, parameters)
	# 	else:
	# 		model = load_model(parameters)
	# 		run_inference(model, parameters)
	# 	# run_batchInference(model, parameters, 'daphneEvalTestData.tsv')
	# else:
	# 	print("Training the model")
	# 	run_trainer(parameters)

	return 0


if __name__ == '__main__':
	main()