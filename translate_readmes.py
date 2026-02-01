import os
import re

def get_translation_map():
    return {
        "# Boosting your Trading Strategy": "# Apni Trading Strategy ko Boost Karein",
        "# Content": "# Vishay-suchi (Content)",
        "## Getting started: adaptive boosting": "## Shuruat: Adaptive Boosting",
        "## The AdaBoost algorithm": "## AdaBoost Algorithm",
        "## Gradient boosting - ensembles for most tasks": "## Gradient Boosting - Adhiktar tasks ke liye ensembles",
        "## How to train and tune GBM models": "## GBM Models ko kaise train aur tune karein",
        "## Using XGBoost, LightGBM and CatBoost": "## XGBoost, LightGBM aur CatBoost ka upyog",
        "## Code Example: A long-short trading strategy with gradient boosting": "## Code Example: Gradient boosting ke saath Long-Short Trading Strategy",
        "## A peek into the black box: How to interpret GBM results": "## Black box mein ek jhalak: GBM results ko kaise interpret karein",
        "## An intraday strategy with Algoseek and LightGBM": "## Algoseek aur LightGBM ke saath ek Intraday Strategy",
        "## Resources": "## Sansadhan (Resources)",
        "This chapter explores": "Yeh chapter explore karta hai",
        "In this chapter, we will see": "Is chapter mein, hum dekhenge",
        "In this chapter, we will cover": "Is chapter mein, hum cover karenge",
        "The notebook": "Notebook",
        "Code Example:": "Code Example:",
        "For example": "Udaharan ke liye",
        "As part of": "Ke hisse ke roop mein",
        "The following sections": "Niche diye gaye sections",
        "Understanding why": "Yeh samajhna ki kyun",
        "Over the last few years": "Pichle kuch saalon mein",
        "feature importance": "feature importance",
        "partial dependence plots": "partial dependence plots",
        "SHAP values": "SHAP values",
        "intraday features": "intraday features",
        "trading signals": "trading signals",
        "out-of-sample predictions": "out-of-sample predictions",
        "backtesting": "backtesting",
        "machine learning algorithms": "machine learning algorithms",
        "decision trees": "decision trees",
        "random forests": "random forests",
        "gradient boosting": "gradient boosting",
        "ensemble learning": "ensemble learning",
        "training data": "training data",
        "test set": "test set",
        "cross-validation": "cross-validation",
        "hyperparameters": "hyperparameters",
        "model": "model",
        "algorithm": "algorithm",
        "implementation": "implementation",
        "library": "library",
        "class": "class",
        "function": "function",
        "variable": "variable",
        "prediction": "prediction",
        "results": "results",
        "performance": "performance",
        "accuracy": "accuracy",
        "error": "error",
        "bias": "bias",
        "variance": "variance",
        "overfitting": "overfitting",
        "underfitting": "underfitting",
        "regularization": "regularization",
        "optimization": "optimization",
        "loss function": "loss function",
        "gradient": "gradient",
        "weights": "weights",
        "samples": "samples",
        "observations": "observations",
        "features": "features",
        "target": "target",
        "output": "output",
        "input": "input"
    }

def deep_hinglish_translate(text):
    # Apply direct map replacements first
    translation_map = get_translation_map()
    for eng, hin in translation_map.items():
        if eng in text:
             text = text.replace(eng, hin)

    # Regex based simple grammatical substitutions for conversational Hinglish
    
    # "is" -> "hai" (simplified)
    text = re.sub(r'\bis\b', 'hai', text, flags=re.IGNORECASE)
    # "in" -> "mein"
    text = re.sub(r'\bin\b', 'mein', text, flags=re.IGNORECASE)
    # "of" -> "ka" (simplified, could be ki/ke but ka works generally)
    text = re.sub(r'\bof\b', 'ka', text, flags=re.IGNORECASE)
    # "and" -> "aur"
    text = re.sub(r'\band\b', 'aur', text, flags=re.IGNORECASE)
    # "to" -> "ko" (or ke liye depending on context, keeping it simple or skipping if risky)
    # text = re.sub(r'\bto\b', 'ko', text, flags=re.IGNORECASE) # risky
    # "we" -> "hum"
    text = re.sub(r'\bwe\b', 'hum', text, flags=re.IGNORECASE)
    # "are" -> "hain"
    text = re.sub(r'\bare\b', 'hain', text, flags=re.IGNORECASE)
    
    # "use" -> "use karte hain" (if verb)
    # This is tricky without NLP, but simple replacement might works for "we use"
    text = re.sub(r'\bwe use\b', 'hum use karte hain', text, flags=re.IGNORECASE)
    text = re.sub(r'\buses\b', 'use karta hai', text, flags=re.IGNORECASE)
    text = re.sub(r'\busing\b', 'use karke', text, flags=re.IGNORECASE)
    
    # "demonstrates"
    text = re.sub(r'\bdemonstrates\b', 'demonstrate karta hai', text, flags=re.IGNORECASE)
    # "provides"
    text = re.sub(r'\bprovides\b', 'provide karta hai', text, flags=re.IGNORECASE)
    # "contains"
    text = re.sub(r'\bcontains\b', 'contain karta hai', text, flags=re.IGNORECASE)
    # "creates"
    text = re.sub(r'\bcreates\b', 'create karta hai', text, flags=re.IGNORECASE)
    
    # "with"
    text = re.sub(r'\bwith\b', 'ke saath', text, flags=re.IGNORECASE)
    
    # "for"
    text = re.sub(r'\bfor\b', 'ke liye', text, flags=re.IGNORECASE)
    
    # "This" -> "Yeh" (at start of sentence mostly)
    text = re.sub(r'^This\b', 'Yeh', text)
    text = re.sub(r'\. This\b', '. Yeh', text)

    return text

def translate_readme(file_path):
    print(f"Translating {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        translated_lines = []
        
        for line in lines:
            if line.strip() == "":
                translated_lines.append(line)
                continue
                
            # Don't translate links too aggressively or code blocks
            if line.startswith('```') or line.startswith('    '):
                translated_lines.append(line)
                continue

            translated_line = deep_hinglish_translate(line)
            translated_lines.append(translated_line)
            
        new_content = '\n'.join(translated_lines)
        
        output_path = file_path.replace('README.md', 'README_HINDI.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Created {output_path}")
        
    except Exception as e:
        print(f"Error translating {file_path}: {e}")

def main():
    root_dir = os.getcwd() # Should be d:\machine-learning-for-trading-main
    print(f"Scanning {root_dir} for README.md files to translate...")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and venv
        if '.git' in dirpath or 'venv' in dirpath or '.ipynb_checkpoints' in dirpath:
            continue
            
        if 'README.md' in filenames:
            readme_path = os.path.join(dirpath, 'README.md')
            # Check if it corresponds to a chapter directory (usually starts with a number)
            # or is specifically Chapter 12
            folder_name = os.path.basename(dirpath)
            
            # Translate if it's a chapter folder or if it's missing a HINDI version
            # Generally safe to generate README_HINDI.md for all relevant folders
            if folder_name.startswith(('0','1','2')) or folder_name == '12_gradient_boosting_machines': 
                 translate_readme(readme_path)

if __name__ == "__main__":
    main()
