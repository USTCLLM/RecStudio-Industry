import importlib

    

def get_modules(module_type: str, module_name: str):
    assert module_type in ["loss", "sampler", "encoder", "interaction", "score", "module"], f"{module_type} is not a valid module type"
    # import the module {module_name} from "rs4industry.model.{module_type}"
    try:
        # from "rs4industry.model.{module_type}" import {module_name}
        module = importlib.import_module(f"rs4industry.model.{module_type}")
        cls = getattr(module, module_name)
        # module = importlib.import_module(module_name, package=pkg)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {module_name} from rs4industry.model.{module_type}") from e
    

def get_model_cls(model_type: str, model_name: str):
    assert model_type in ["retriever", "ranker"], f"{model_type} is not a valid model type"
    try:
        module = importlib.import_module(f"rs4industry.model.{model_type}s")
        cls = getattr(module, model_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {model_name} from rs4industry.model.{model_type}s") from e