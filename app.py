from flask import Flask, request, jsonify
from transformers import pipeline
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the pipeline when the app starts
# This will load the model into memory once
try:
    logger.info("Loading StarCoder2-3B model... This may take a few minutes.")
    pipe = pipeline(
        "text-generation", 
        model="bigcode/starcoder2-3b",
        device_map="auto",  # Automatically uses GPU if available
        torch_dtype="auto",
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    pipe = None

@app.route('/generate', methods=['POST'])
def generate_code():
    # Check if model was loaded successfully
    if pipe is None:
        return jsonify({"error": "Model not available"}), 500
    
    # Get JSON data from request
    data = request.get_json()
    
    # Validate request
    if not data or 'prompt' not in data:
        return jsonify({"error": "Please provide a 'prompt' in the JSON body"}), 400
    
    prompt = data['prompt']
    
    # Optional parameters with defaults
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    do_sample = data.get('do_sample', True)
    
    try:
        logger.info(f"Generating code for prompt: {prompt[:100]}...")
        
        # Generate code
        result = pipe(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=pipe.tokenizer.eos_token_id  # Important for code models
        )
        
        generated_text = result[0]['generated_text']
        
        # Extract only the new generated part (optional)
        # generated_part = generated_text[len(prompt):]
        
        logger.info("Generation completed successfully")
        
        return jsonify({
            "prompt": prompt,
            "generated_code": generated_text,  # or use generated_part
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running and model is loaded"""
    status = "ready" if pipe is not None else "not ready"
    return jsonify({"status": status, "model": "starcoder2-3b"})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
