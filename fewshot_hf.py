from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

checkpoint = "bigscience/bloomz"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

dataset = load_dataset("boolq")


# generate a prompt containing 8 examples
true_idx = []
false_idx = []
i = 0
while True:
	if len(true_idx) < 4:
		if dataset['train'][i]['answer'] == True:
			true_idx.append(i)
	elif len(false_idx) < 4:
		if dataset['train'][i]['answer'] == False:
			false_idx.append(i)
	else:
		break
	i += 1

prompt = ""
for tidx, fidx in zip(true_idx, false_idx):
	prompt += "passage:" + dataset['train'][tidx]['passage'] + "\n"
	prompt += "question:" + dataset['train'][tidx]['question'] + "\n"
	prompt += "answer:" + str(dataset['train'][tidx]['answer']) + "\n"
	prompt += "\n"

	prompt += "passage:" + dataset['train'][fidx]['passage'] + "\n"
	prompt += "question:" + dataset['train'][fidx]['question'] + "\n"
	prompt += "answer:" + str(dataset['train'][fidx]['answer']) + "\n"
	prompt += "\n"

# print(prompt)


# evaluate
true_idx = []
false_idx = []
i = 0
while True:
	if len(true_idx) < 15:
		if dataset['validation'][i]['answer'] == True:
			true_idx.append((i, True))
	elif len(false_idx) < 15:
		if dataset['validation'][i]['answer'] == False:
			false_idx.append((i, False))
	else:
		break
	i += 1

idx = true_idx + false_idx


num_correct = 0

for i, label in idx:
	new_prompt = "passage:" + dataset['validation'][i]['passage'] + "\n"
	new_prompt += "question:" + dataset['validation'][i]['question'] + "\n"
	new_prompt += "answer:"

	inputs = tokenizer.encode(prompt + new_prompt, return_tensors="pt")
	outputs = model.generate(inputs)
	pred = tokenizer.decode(outputs[0])
	print(pred, label)


# 	print("-------", pred, str(label), pred == str(label), "-------")
# 	if pred == str(label):
# 		num_correct += 1
#
# print(num_correct / len(idx)) # 0.8666666666666667

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))