import os
from llama_cpp import Llama

import comfy.model_management as mm
import comfy.utils
import folder_paths
llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']

script_directory = os.path.dirname(os.path.abspath(__file__))

folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)

class llama_cpp_model_loader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "n_gpu_layers": ("INT", {"default": 0, "min": -1, "max": 4096, "step": 1}),
            "download_default": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            "model": (folder_paths.get_filename_list("LLM"),),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llamamodel",)
    FUNCTION = "loadmodel"
    CATEGORY = "Llama-cpp"

    def loadmodel(self, n_gpu_layers, download_default, model=None):
        mm.soft_empty_cache()
        
        custom_config = {
            "model": model,
        }
        if not hasattr(self, "model") or custom_config != self.current_config:
            self.current_config = custom_config
            llama3_dir = (os.path.join(folder_paths.models_dir, 'LLM', 'llama3'))
            default_checkpoint_path = os.path.join(llama3_dir, 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
            default_model = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            if download_default and not os.path.exists(default_checkpoint_path):
                print(f"Downloading {default_model}")
                from huggingface_hub import snapshot_download
                allow_patterns = [f"*{default_model}*"]
                snapshot_download(repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF", 
                                  allow_patterns=allow_patterns, 
                                  local_dir=llama3_dir, 
                                  local_dir_use_symlinks=False
                                  )
                model_path = default_checkpoint_path
            else:
                model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
            print(f"Loading model from {model_path}")

            llm = Llama(model_path, n_gpu_layers=n_gpu_layers)
   
        return (llm,)
            
class llama_cpp_instruct:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "llmamamodel": ("LLAMACPPMODEL",),
            "parameters": ("LLAMACPPARAMS", ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "prompt": ("STRING", {"multiline": True, "default": "How much wood would woodchuck chuck, if woodchuck could chuck wood?",}),

            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "Llama-cpp"

    def process(self, llmamamodel, prompt, parameters, seed):
        
        mm.soft_empty_cache()
        output = llmamamodel(
        f"Q: {prompt} A: ", # Prompt
        stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        echo=False,
        seed=seed,
        max_tokens = parameters.get("max_tokens", 32),
        top_k = parameters.get("top_k", 40),
        top_p = parameters.get("top_p", 0.95),
        min_p = parameters.get("min_p", 0.05),
        typical_p = parameters.get("typical_p", 1.0),
        temperature = parameters.get("temperature", 0.8),
        repeat_penalty = parameters.get("repeat_penalty", 1.1),
        frequency_penalty = parameters.get("frequency_penalty", 0.0),
        presence_penalty = parameters.get("presence_penalty", 0.0),
        tfs_z = parameters.get("tfs_z", 1.0),
        mirostat_mode = parameters.get("mirostat_mode", 0),
        mirostat_eta = parameters.get("mirostat_eta", 0.1),
        mirostat_tau = parameters.get("mirostat_tau", 5.0),
        )
        print(output)
        text = output['choices'][0]['text']
        return (text,)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "max_tokens": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                }        
        }

    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "Llama-cpp"

    def process(self, max_tokens, top_k, top_p, min_p, typical_p, temperature, repeat_penalty, 
                frequency_penalty, presence_penalty, tfs_z, mirostat_mode, mirostat_eta, mirostat_tau, 
                ):
        
        parameters_dict = {
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "typical_p": typical_p,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "tfs_z": tfs_z,
        "mirostat_mode": mirostat_mode,
        "mirostat_eta": mirostat_eta,
        "mirostat_tau": mirostat_tau,
        } 
        return (parameters_dict,)
    
NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct": llama_cpp_instruct,
    "llama_cpp_parameters": llama_cpp_parameters
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct": "Llama-cpp Instruct",
    "llama_cpp_parameters": "Llama-cpp Parameters"
}

