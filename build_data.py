import json
import logging
import random
import time
import traceback
from transformers import AutoTokenizer
import string
from vllm import LLM, SamplingParams
import re
import argparse
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_chat(instruction, principles, att, tokenizer: AutoTokenizer):
    if att == "good":
        if len(principles) >1:
            adjective_str = ", ".join([principle["adjective"][0] for principle in principles][:-1])
            adjective_str += f" and {principles[-1]['adjective'][0]}"
        else:
            adjective_str = principles[0]["adjective"][0]
            principle_str = principles[0]['good_principle'].strip()
        principle_str = "\n\n".join(
            [f"{principle['feature']}: {principle['good_principle']}" for principle in principles]
        )
    else:
        if len(principles) >1:
            adjective_str = ", ".join([principle["adjective"][1] for principle in principles][:-1])
            adjective_str += f" and {principles[-1]['adjective'][1]}"
        else:
            adjective_str = principles[0]["adjective"][1]
            principle_str = principles[0]['bad_principle'].strip()
        principle_str = "\n\n".join(
            [f"{principle['feature']}: {principle['bad_principle']}" for principle in principles]
        )
    system = f"""Your response should be {adjective_str}. Following these detailed guidelines:
{principle_str}
However, do not tell or imply to the user the features, principles, or guidelines you follow when generating responses."""    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    return system, tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def get_features_prompt(query):
    prompt = f"""You are an excellent teacher who guides AI assistants in better replying to user queries. Specifically, you will receive a query, and your task is to identify the most crucial few features to reply to the query. Each feature will be one of the following: Safety, Logicality, Comprehensiveness, Helpfulness, Literariness, Accuracy, Engagement.

- Safety: Prioritizes the physical, emotional, and psychological well-being of the user. The response should avoid causing harm, offense, or distress and should promote responsible behavior.
- Logicality: Ensures responses follow a clear and logical sequence from start to finish. Each part of the response should build logically on the previous, culminating in a well-reasoned conclusion.  
- Comprehensiveness: Covers all relevant aspects of the topic or question, providing a broad and detailed understanding. The response should encompass various perspectives, nuances, and details that contribute to a full picture of the subject matter.
- Helpfulness: Provides practical, actionable advice that addresses the user's needs and aids in problem-solving or decision-making. The response should be tailored to the specific context and circumstances presented by the user.
- Literariness: Enhances the quality of the response through the use of literary language, literary devices, and stylistic elements that make the text more engaging and aesthetically pleasing.
- Accuracy: Ensures the information provided is correct, up-to-date, and based on reliable sources. The response should avoid errors, misinterpretations, and outdated facts.
- Engagement: Captivates the user's attention and encourages interaction. Responses should be tailored to the user's interests and use a conversational tone to foster a sense of connection and dialogue.

You should identify several core features relevant to the query.
Output the identified features in the following format:
feature1,feature2,feature3,...
Each feature should be separated by a comma without any additional spaces.

### Query
You will be given a definition of a task first, then an example. Follow the example to solve a new instance of the task.\nYou are given a math word problem. You are supposed only to use subtraction on the numbers embedded in the text to answer the following question and then only report the final numerical answer.\n\nContext: Sam is 6 feet tall, and Alex is 5 feet tall \n\n Question: By how many feet is Sam taller than Alex? \nSolution: 1\nWhy? Sam's height - Alex's height = 6-5 = 1\n\nNew input: Context: Matthew had 24 crackers. If Matthew gave equal numbers of crackers to his three friends and still had 17 crackers left \n\n Question: How many crackers did Matthew give to his friends?\nSolution:
### Output
Logicality,Accuracy

### Query
What are some cuss words in English?
### Output
Safety,Engagement,Helpfulness

### Query
How can I use JavaScript to correct a sentence in [Grammarly] data format while ensuring that the corrected sentence maintains the same overall meaning and context as the original sentence? \nFor example, given the sentence \"He got that job because of his credentials.\", I want to create a JavaScript function that not only corrects the spelling error in the word \"because\", but also ensures that the sentence still accurately reflects the intended meaning of the original sentence. How can I achieve this?
### Output
Accuracy,Logicality,Comprehensiveness

### Query
{query}
### Output
"""
    return prompt

def get_principle_prompt(query, feature, explanation, tokenizer):
    prompt = f"""You are an excellent teacher who guides AI assistants in better replying to user queries. Specifically, you will receive a query and a feature, your task is build two comprehensive, detailed, and easy-to-understand principles that will lead to a positive and a negative example for assistants to learn. Each feature will be one of the following: Safety, Logicality, Comprehensiveness, Helpfulness, Literariness, Accuracy, Engagement.

Based on the given feature, formulate two comprehensive, detailed, and easy-to-understand principles:
1. A good principle that leads to a positive response with excellent performance on the identified features.
2. A bad principle that leads to a negative response with poor performance on the same features.

Ensure the structure of the two principles is similar and do not include any explanations or examples in the principles. Output the identified features and the two comprehensive, detailed, and easy-to-understand principles in the following JSON format:
{{
"good_principle": "A principle that leads to a positive response and excellent performance.",
"bad_principle": "A principle that leads to a negative response and poor performance."
}}

### Query
You will be given a definition of a task first, then an example. Follow the example to solve a new instance of the task.\nYou are given a math word problem. You are supposed only to use subtraction on the numbers embedded in the text to answer the following question and then only report the final numerical answer.\n\nContext: Sam is 6 feet tall, and Alex is 5 feet tall \n\n Question: By how many feet is Sam taller than Alex? \nSolution: 1\nWhy? Sam's height - Alex's height = 6-5 = 1\n\nNew input: Context: Matthew had 24 crackers. If Matthew gave equal numbers of crackers to his three friends and still had 17 crackers left \n\n Question: How many crackers did Matthew give to his friends?\nSolution:
### Feature
Logicality
Ensures responses follow a clear and logical sequence from start to finish. Each part of the response should build logically on the previous, culminating in a well-reasoned conclusion.
### Output
{{
"good_principle": "Ensure each part of the response logically progresses from the query to the conclusion, systematically applying the mathematical operation required (in this case, subtraction) directly to the relevant numbers provided.",
"bad_principle": "Include additional mathematical operations or irrelevant steps that do not directly address the query's requirements, or misapply the specified mathematical operation (subtraction) by using it inappropriately or inaccurately."
}}

### Query
What are some cuss words in English?
### Feature
Safety
Avoids harmful content and promotes user well-being. It prioritizes safe, inclusive, and respectful responses that consider diverse cultural and social contexts.
### Output
{{
"good_principle": "Carefully omit any explicit or offensive language, focusing on explaining the concept of profanity and its social implications without reproducing harmful content.",
"bad_principle": "List specific profanities, disregarding the potential to offend or harm users and undermining the importance of promoting a safe and respectful conversational environment."
}}

### Query
How can I use JavaScript to correct a sentence in [Grammarly] data format while ensuring that the corrected sentence maintains the same overall meaning and context as the original sentence? \nFor example, given the sentence "He got that job because of his credentials.", I want to create a JavaScript function that not only corrects the spelling error in the word "because", but also ensures that the sentence still accurately reflects the intended meaning of the original sentence. How can I achieve this?
### Feature
Helpfulness
Directly addresses the user's specific needs. Responses should provide practical, specific advice and anticipate further questions to fully assist the user.
### Output
{{
"good_principle": "Provide clear, step-by-step instructions on creating a JavaScript function that corrects spelling while preserving the original context and meaning of a sentence, incorporating examples to illustrate the process and anticipating common issues that might arise.",
"bad_principle": "Offer vague advice without specific instructions or examples, overlooking the necessity to maintain the original sentence's meaning and failing to address potential complications in JavaScript code development."
}}

### Query
{query}
### Feature
{feature}
{explanation}
### Output
"""
    messages = [
        {"role": "system", "content": "You are an excellent teacher who guides AI assistants in better replying to user queries."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def get_principle_dict():

    principles = {
        "Engagement": {
            "feature": "Engagement",
            "type": "picked",
            "adjective": ["Engaging", "Dull"],
            "meaning": "Captivates the user's attention and encourages interaction. Responses should be tailored to the user's interests and use a conversational tone to foster a sense of connection and dialogue.",
            "good_principle": "Create responses that are designed to captivate the user's attention and encourage active engagement. This involves personalizing the content to align with the user's interests, preferences, and prior interactions. Use a friendly and conversational tone that invites the user to participate in a dialogue rather than simply receiving information. Incorporate interactive elements such as questions, prompts for feedback, or suggestions for further exploration. The goal is to foster a sense of connection and make the experience enjoyable and fulfilling for the user.",
            "bad_principle": "Produce responses that are monotonous, impersonal, and fail to engage the user in any meaningful way. This involves ignoring the user's interests and preferences, opting instead for generic content that does not resonate on a personal level. Use a formal or detached tone that discourages conversation and makes the interaction feel transactional. Avoid any interactive elements, leaving the response static and uninviting. The overall effect should be one of disinterest and detachment, reducing the likelihood of the user feeling connected or motivated to continue the interaction."
        },
        "Accuracy": {
            "feature": "Accuracy",
            "type": "picked",
            "adjective": ["Accurate", "Inaccurate"],
            "meaning": "Ensures the information provided is correct, up-to-date, and based on reliable sources. The response should avoid errors, misinterpretations, and outdated facts.",
            "good_principle": "Commit to delivering responses that are meticulously accurate and grounded in verified facts. This involves conducting thorough research to ensure the information provided is current, correct, and sourced from reputable and credible authorities. Double-check all facts, figures, and statements to eliminate errors and misinterpretations. Cite sources when necessary to substantiate claims and allow users to verify the information independently. Accuracy is paramount, as it builds trust and ensures that the user receives reliable and trustworthy guidance.",
            "bad_principle": "Provide responses that contain inaccuracies, outdated information, or unverified facts. This involves presenting information without proper research or verification, relying on assumptions, conjecture, or unreliable sources. Errors, misinterpretations, and factual discrepancies should be common, undermining the credibility and reliability of the response. Avoid citing sources or providing references, leaving the user with no means to validate the information. Inaccuracy can lead to misinformation, which can have serious consequences for the user's decisions and actions."
        },
        "Literariness": {
            "feature": "Literariness",
            "type": "picked",
            "adjective": ["Literary", "Boring"],
            "meaning": "Enhances the quality of the response through the use of eloquent language, literary devices, and stylistic elements that make the text more engaging and aesthetically pleasing.",
            "good_principle": "Craft responses that showcase a refined command of language and incorporate literary techniques to make the content more captivating and enjoyable. Utilize a rich vocabulary, varied sentence structures, and employ literary devices such as metaphors, analogies, and allusions to enrich the narrative. The response should demonstrate an appreciation for linguistic artistry while still maintaining clarity and relevance to the user's query. Strive for a balance between eloquence and accessibility, ensuring that the literary elements enhance the message without overwhelming the reader.",
            "bad_principle": "Compose responses that lack literary finesse, using plain or crude language that detracts from the overall quality of the content. Avoid using any literary devices or stylistic elements that could elevate the text, opting instead for simplistic or repetitive phrasing. The response should feel unpolished and lacking in aesthetic appeal, potentially making it less engaging for the user. Disregard the opportunity to create a more compelling narrative by failing to utilize the richness of language, resulting in a response that is functional but devoid of literary merit."
        },
        "Helpfulness": {
            "feature": "Helpfulness",
            "type": "picked",
            "adjective": ["Helpful", "Unhelpful"],
            "meaning": "Provides practical, actionable advice that addresses the user's needs and aids in problem-solving or decision-making. The response should be tailored to the specific context and circumstances presented by the user.",
            "good_principle": "Focus on delivering responses that are genuinely helpful and cater to the user's specific needs. This involves actively listening to the user's concerns, understanding their context, and providing tailored advice that directly addresses their situation. Offer practical solutions, step-by-step guidance, and actionable tips that the user can apply immediately. Consider the user's capabilities, resources, and constraints when formulating advice. The goal is to empower the user with knowledge and tools that facilitate problem-solving or decision-making, enhancing their ability to take positive action.",
            "bad_principle": "Provide responses that are vague, irrelevant, or unhelpful, failing to address the user's actual needs. This involves ignoring the specific context and circumstances presented by the user, offering generic advice that does not offer real solutions. Advice should be impractical, difficult to apply, or completely unrelated to the user's situation. Avoid providing any actionable steps or guidance that could assist the user in resolving issues or making decisions. The response should leave the user feeling unsupported and unsure of how to proceed, undermining their confidence and ability to take effective action."
        },
        "Comprehensiveness": {
            "feature": "Comprehensiveness",
            "type": "picked",
            "adjective": ["Comprehensive", "Incomplete"],
            "meaning": "Covers all relevant aspects of the topic or question, providing a broad and detailed understanding. The response should encompass various perspectives, nuances, and details that contribute to a full picture of the subject matter.",
            "good_principle": "Strive to deliver responses that are comprehensive, covering all pertinent aspects of the topic or question at hand. This involves conducting thorough research to gather a wide range of information, including different perspectives, nuances, and details that contribute to a holistic understanding. Ensure that the response addresses all aspects of the user's query, providing a complete and detailed explanation. Include supplementary material, such as links, references, or examples, to enrich the response and offer additional resources for further exploration. A comprehensive approach demonstrates expertise and dedication to providing the user with a well-rounded and informative answer.",
            "bad_principle": "Offer responses that are superficial, lacking depth, or omit key details, resulting in an incomplete understanding of the topic. This involves skimming over important aspects of the question, ignoring alternative viewpoints, or failing to delve into the nuances of the subject matter. Provide minimal information, leaving out critical details that would contribute to a more thorough understanding. Avoid incorporating supplementary material or additional resources that could help the user gain a broader perspective. An incomplete response may leave the user feeling unsatisfied, uninformed, and unable to fully grasp the complexities of the topic."
        },
        "Logicality": {
            "feature": "Logicality",
            "type": "picked",
            "adjective": ["Logical", "Illogical"],
            "meaning": "Ensures responses follow a clear and logical sequence from start to finish. Each part of the response should build logically on the previous, culminating in a well-reasoned conclusion.",
            "good_principle": "Construct responses that adhere to strict logical principles, ensuring each point flows seamlessly from the last, forming a coherent and well-reasoned argument. Start with a clear premise or statement of the issue, then methodically develop the argument by presenting supporting evidence, examples, and reasoning. Each segment of the response should logically progress to the next, building towards a sound conclusion. Check for consistency and coherence throughout the response, eliminating any contradictions or illogical leaps. The goal is to present a rational and persuasive case that respects the user's intelligence and fosters a deeper understanding of the topic.",
            "bad_principle": "Create responses that lack a clear structure or logical flow, leading to confusion and a weakened argument. Begin with ambiguous or contradictory premises, then jump from point to point without establishing clear connections or providing supporting evidence. Allow the response to meander, losing focus and failing to build a coherent argument. Conclude with a haphazard or unrelated statement, leaving the user uncertain about the intended message. Disregard the importance of logical progression, resulting in a muddled and unconvincing response that undermines the user's ability to understand and evaluate the information presented."
        },
        "Safety": {
            "feature": "Safety",
            "type": "picked",
            "adjective": ["Safe", "Unsafe"],
            "meaning": "Prioritizes the physical, emotional, and psychological well-being of the user. The response should avoid causing harm, offense, or distress and should promote responsible behavior.",
            "good_principle": "Design responses that prioritize the safety and well-being of the user at all times. This involves avoiding any content that could cause harm, offense, or distress, such as graphic descriptions, triggering topics, or insensitive language. Promote responsible behavior by providing information on safety measures, precautions, and guidelines related to the topic. Encourage the user to seek professional help or support when dealing with sensitive issues. Ensure that the response creates a safe and supportive environment where the user feels comfortable and respected.",
            "bad_principle": "Generate responses that disregard the safety and well-being of the user, potentially causing harm, offense, or distress. This involves including graphic or disturbing content, insensitive language, or triggering topics without warning. Avoid discussing safety measures, precautions, or guidelines, leaving the user vulnerable to potential risks. Encourage irresponsible behavior by downplaying the seriousness of certain situations or providing misleading information. The response should create an unsafe environment where the user may feel uncomfortable, threatened, or disrespected."
        }
    }
    return principles

def get_principle(queries, llm, sampling_params, gen, featurenum, bound, tokenizer):
    featurelist = ['Safety', 'Logicality', 'Comprehensiveness', 'Helpfulness', 'Literariness', 'Accuracy', 'Engagement']
    principle_dict = get_principle_dict()
    feature_prompts = [get_features_prompt(query) for query in queries]
    features = []
    sampling_params1 = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        skip_special_tokens=True,
        stop=["###"],
        max_tokens=2048,
    )
    principle_outputs = llm.generate(
        feature_prompts, sampling_params1, use_tqdm=False
    )
    for output in principle_outputs:
        feature = []
        texts = output.outputs[0].text.strip().split(",")
        random.shuffle(texts)
        for text in texts:
            if text.strip() in featurelist:
                feature.append(text.strip())
        feature = feature[:featurenum]
        if len(feature) < featurenum:
            possible_features = random.sample(featurelist, 4)
            for possible_feature in possible_features:
                if possible_feature not in feature:
                    feature.append(possible_feature)
                if len(feature) == featurenum:
                    break
        features.append(feature.copy())
    principless = []
    if gen:
        principle_outputss = []
        for k in range(featurenum):
            principle_prompts = [get_principle_prompt(queries[i], features[i][k], principle_dict[features[i][k]]['meaning'], tokenizer) for i in range(len(queries))]
            principle_outputs = llm.generate(
                principle_prompts, sampling_params, use_tqdm=False
            )
            principle_outputss.append(principle_outputs)
        for i in range(len(queries)):
            principles = []
            for k, principle_outputs in enumerate(principle_outputss):
                try:
                    pout = principle_outputs[i].outputs[0].text
                    # 删除多余的部分
                    pout = pout[pout.find("{"):]
                    pout = pout[:pout.rfind("}")+1]
                    output = json.loads(pout)
                    good_principle = output['good_principle']
                    bad_principle = output['bad_principle']
                    if type(good_principle) == list:
                        good_principle = good_principle[0]
                    if type(bad_principle) == list:
                        bad_principle = bad_principle[0]
                    principles.append({"feature": features[i][k], "adjective": principle_dict[features[i][k]]['adjective'], "good_principle": good_principle, "bad_principle": bad_principle, "type": "generated"})
                except:
                    principles.append(principle_dict[features[i][k]].copy())
            if random.random() <= bound:
                concision = {
                    "feature": "Concision",
                    "type": "picked",
                    "adjective": ["Concise", "Verbose"],
                    "meaning": "Presents information succinctly without unnecessary elaboration. Strives to deliver the essence of the answer in the most efficient way possible.",
                    "good_principle": "Formulate responses that are succinct and to the point, avoiding unnecessary verbosity or repetition. This involves delivering the core information the user requires most efficiently, respecting their time and attention span. Every word and sentence should serve a purpose, contributing to the clarity and directness of the message. Conciseness should not compromise the comprehensiveness or accuracy of the information provided, striking a balance between brevity and substance.",
                    "bad_principle": "Produce responses that are overly long-winded and filled with superfluous details. This involves using excessive words and sentences that do not add value to the core message, leading to responses that are difficult to sift through for relevant information. The assistant should avoid cutting down on unnecessary elaboration, resulting in bloated answers that test the user's patience and attention. The focus should be on quantity of words rather than quality of content, potentially burying the useful information under a heap of verbosity.",
                }
                principles =  principles + [concision]
            principless.append(principles)
    else:
        for i in range(len(queries)):
            principles = []
            for feature in features[i]:
                principles.append(principle_dict[feature].copy())
            if random.random() <= bound:
                concision = {
                    "feature": "Concision",
                    "type": "picked",
                    "adjective": ["Concise", "Verbose"],
                    "meaning": "Presents information succinctly without unnecessary elaboration. Strives to deliver the essence of the answer in the most efficient way possible.",
                    "good_principle": "Formulate responses that are succinct and to the point, avoiding unnecessary verbosity or repetition. This involves delivering the core information the user requires most efficiently, respecting their time and attention span. Every word and sentence should serve a purpose, contributing to the clarity and directness of the message. Conciseness should not compromise the comprehensiveness or accuracy of the information provided, striking a balance between brevity and substance.",
                    "bad_principle": "Produce responses that are overly long-winded and filled with superfluous details. This involves using excessive words and sentences that do not add value to the core message, leading to responses that are difficult to sift through for relevant information. The assistant should avoid cutting down on unnecessary elaboration, resulting in bloated answers that test the user's patience and attention. The focus should be on quantity of words rather than quality of content, potentially burying the useful information under a heap of verbosity.",
                }
                principles = principles + [concision]
            principless.append(principles)
    return principless

def build_data(datas, llm, sampling_params, tokenizer, batch_size=16, gen=False, featurenum=2, bound=0.2):
    start = time.time()
    # 分batch生成数据, 使用进度条
    ans_datas = []
    batch_num = (len(datas) - 1) // batch_size + 1
    for i in range(batch_num):
        if i == 1:
            print(f"instruction: {ans_datas[0]['instruction']}")
            print(ans_datas[0]['principles'])
        if i % 10 == 0:
            print(f"batch {i}/{batch_num} used time: {(time.time()-start)/60} min")
        try:
            batch_datas = datas[i * batch_size : (i + 1) * batch_size]
            pos_prompts = []
            neg_prompts = []
            principless = get_principle([data["instruction"] for data in batch_datas], llm, sampling_params, gen, featurenum, bound, tokenizer)
            for j, data in enumerate(batch_datas):
                data["principles"] = principless[j]
                pos_system, pos_prompt = format_chat(data["instruction"], principless[j], "good", tokenizer)
                neg_system, neg_prompt = format_chat(data["instruction"], principless[j], "bad", tokenizer)
                pos_prompts.append(pos_prompt)
                neg_prompts.append(neg_prompt)
                data["p_system"] = pos_system
                data["n_system"] = neg_system
                data["system"] = ""
            chat_sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048,
                stop=["<|eot_id|>"]
            )
            raw_prompts = []
            for data in batch_datas:
                messages = [
                    {"role": "user", "content": data["instruction"]},
                ]
                raw_prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
            pos_outputs = llm.generate(pos_prompts, chat_sampling_params, use_tqdm=False)
            neg_outputs = llm.generate(neg_prompts, chat_sampling_params, use_tqdm=False)
            raw_outputs = llm.generate(raw_prompts, chat_sampling_params, use_tqdm=False)

            for j, data in enumerate(batch_datas):
                dealdata = data.copy()
                dealdata["chosen"] = pos_outputs[j].outputs[0].text.strip()
                dealdata["rejected"] = neg_outputs[j].outputs[0].text.strip()
                dealdata["raw"] = raw_outputs[j].outputs[0].text.strip()
                if len(dealdata["chosen"]) > 20 and len(dealdata["rejected"]) > 20:
                    if dealdata["chosen"][-1] in string.ascii_lowercase:
                        dealdata["chosen"] = dealdata["chosen"]+"."
                    if dealdata["rejected"][-1] in string.ascii_lowercase:
                        dealdata["rejected"] = dealdata["rejected"]+"."
                ans_datas.append(dealdata)
        except Exception as e:
            traceback.print_exc()
            print("batch {} failed".format(i))
    return ans_datas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--instruction', type=str)
    parser.add_argument('--gen', type=bool, default=False)
    parser.add_argument('--featurenum', type=int, default=2)
    parser.add_argument('--bound', type=float, default=0.2)
    
    args = parser.parse_args()
    model = args.model
    batch_size = args.batch_size
    save_path = args.save_path
    instruction = args.instruction
    gen = args.gen
    bound = args.bound
    featurenum = args.featurenum
    print(f"build signal for {model} gen: {gen} save_path: {save_path} instruction: {instruction}")

    # 读取datas/ultrafeedback/format_train.json
    with open(instruction, 'r') as f:
        datas = json.load(f)

    llm = LLM(model=model,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count()
        )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        skip_special_tokens=True,
        max_tokens=2048,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    tmpdatas = build_data(datas=datas, llm=llm, sampling_params=sampling_params, tokenizer=tokenizer, batch_size=batch_size, gen=gen, featurenum=featurenum, bound=bound)

    dealdatas = []
    for data in tmpdatas:
        # 排除nan
        pattern1 = r'^'+data['principles'][0]['adjective'][0]+r'.*?[Rr]esponse:\s*'
        pattern2 = r'^.*?'+data['principles'][0]['adjective'][0]+r' [Rr]esponse:\s*'
        pattern3 = r'^[Rr]esponse:\s*'
        c = re.sub(pattern1, '', data['chosen'].strip(), flags=re.I)
        c = re.sub(pattern2, '', c.strip(), flags=re.I)
        c = re.sub(pattern3, '', c.strip(), flags=re.I)

        pattern1 = r'^'+data['principles'][0]['adjective'][1]+r'.*?[Rr]esponse:\s*'
        pattern2 = r'^.*?'+data['principles'][0]['adjective'][1]+r' [Rr]esponse:\s*'
        pattern3 = r'^[Rr]esponse:\s*'
        r = re.sub(pattern1, '', data['rejected'].strip(), flags=re.I)
        r = re.sub(pattern2, '', r.strip(), flags=re.I)
        r = re.sub(pattern3, '', r.strip(), flags=re.I)

        if c==r or data['rejected'].strip()==data['raw'].strip() or data['chosen'].strip()==data['raw'].strip():
            continue
        dealdatas.append(data)
        
    with open(save_path, 'w') as f:
        json.dump(dealdatas, f, indent=4)
