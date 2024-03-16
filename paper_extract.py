from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class llama2_chain:

    def __init__(self, model_path, n_gpu_layers, n_batch):
        self.model_path=model_path
        self.n_gpu_layers=n_gpu_layers
        self.n_batch = n_batch
    
    def llm_set(self):
        metadata_extraction_template = """
        Given the following Actual Text, extract the title, authors, and abstract. 
        and do not say anything else.

        Actual Text: {text}

        """

        prompt_template = PromptTemplate(template=metadata_extraction_template, input_variables=["text"])

        #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        callback_manager = CallbackManager([])

        llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            callback_manager=callback_manager,
            verbose=False,
            n_ctx=3000
        )

        llm_chain = LLMChain(prompt=prompt_template, llm=llm)

        input_text = """
        Autoencoders
        Dor Bank, Noam Koenigstein, Raja Giryes
        Abstract An autoencoder is a specific type of a neural network, which is mainly
        designed to encode the input into a compressed and meaningful representation, and
        then decode it back such that the reconstructed input is similar as possible to the
        original one. This chapter surveys the different types of autoencoders that are mainly
        used today. It also describes various applications and use-cases of autoencoders.
        """

        prompt = prompt_template.format(text=input_text)
        return llm_chain, prompt
    

model_paths = ["C:/Users/kbh/Code/models/llama-2-7b-chat.Q2_K.gguf",
               "C:/Users/kbh/Code/models/llama-2-13b-chat.Q3_K_M.gguf",
               "C:/Users/kbh/Code/models/llama-2-13b-chat.Q4_K_M.gguf",
               "C:/Users/kbh/Code/models/llama-2-13b-chat.Q5_K_M.gguf",
               "C:/Users/kbh/Code/models/llama-2-13b-chat.Q8_0.gguf"]
n_batches = [2000, 3000, 4000]
n_gpu = 50
import time

f=open('C:/Users/kbh/Code/expr.txt', 'w')

for path in model_paths:
    for n_batch in n_batches:
        llm = llama2_chain(path, n_batch=n_batch, n_gpu_layers=n_gpu)
        llm_chain, prompt = llm.llm_set()
        cnt=0
        avg_time = 0
        for i in range(100):
            start_time = time.time()
            response = llm_chain.invoke(prompt)
            end_time = time.time()
            if response['text'].find('Title') != -1 and response['text'].find('Authors') and response['text'].find('Abstract'):
                cnt+=1
            if i!=0:
                curr_time = end_time-start_time
                avg_time += curr_time
        f.write('===============================\n')
        f.write('model : %s\n' %path)
        f.write('batch : %d\n' %n_batch)
        f.write('acc : %d%%\n' %cnt)
        f.write('time : %.2f\n' %avg_time)
        f.write('===============================\n\n')
    

