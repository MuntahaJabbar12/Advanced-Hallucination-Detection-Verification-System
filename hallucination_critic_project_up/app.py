from flask import Flask, render_template, request, jsonify
from hallucination_detector import HallucinationCritic
import json

app = Flask(__name__)
critic = HallucinationCritic()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze source text and summary for hallucinations.
    
    Expects JSON with:
    - source_text: Original source text
    - summary_text: AI-generated summary
    
    Returns:
    - JSON with analysis results
    """
    try:
        data = request.get_json()
        source_text = data.get('source_text', '').strip()
        summary_text = data.get('summary_text', '').strip()
        
        if not source_text or not summary_text:
            return jsonify({
                'error': 'Both source text and summary text are required'
            }), 400
        
        # Perform analysis
        report = critic.generate_hallucination_report(source_text, summary_text)
        
        # Convert sets to lists for JSON serialization
        result = {
            'verdict': report['verdict'],
            'hallucination_score': float(report['hallucination_score']),
            'semantic_similarity': float(report['semantic_similarity']),
            'entity_analysis': {
                'hallucinated': report['entity_analysis']['hallucinated'],
                'verified': report['entity_analysis']['verified'],
                'source_entities': report['entity_analysis']['source_entities'],
            },
            'numerical_analysis': {
                'hallucinated': report['numerical_analysis']['hallucinated'],
                'verified': report['numerical_analysis']['verified'],
                'source_numbers': report['numerical_analysis']['source_numbers'],
            },
            'keyword_analysis': {
                'coverage_rate': float(report['keyword_analysis']['coverage_rate']),
                'covered': list(report['keyword_analysis']['covered']) if isinstance(report['keyword_analysis']['covered'], set) else report['keyword_analysis']['covered'],
                'missing': list(report['keyword_analysis']['missing']) if isinstance(report['keyword_analysis']['missing'], set) else report['keyword_analysis']['missing'],
            },
            'explanation': report['explanation'],
            'computational_trace': report['computational_trace'],
            'false_reasons': report['false_reasons'],
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/examples')
def get_examples():
    """Return example texts for testing"""
    examples = {
        'clean': {
            'source': """Apple Inc. announced its Q4 2023 earnings on November 2, 2023. The company reported revenue of $89.5 billion, a 1% decrease from the previous year. CEO Tim Cook stated that iPhone sales remained strong despite market challenges. The company's services division grew by 16% year-over-year. Apple's wearables and accessories segment also showed positive growth.""",
            'summary': """Apple reported Q4 2023 earnings with revenue of $89.5 billion on November 2. iPhone sales were strong according to CEO Tim Cook, and services grew 16% year-over-year."""
        },
        'hallucinated': {
            'source': """Tesla released its Cybertruck in November 2023. The electric pickup truck features a stainless steel exoskeleton and starts at $60,990. Initial production will be limited to 250,000 units per year. The vehicle can tow up to 11,000 pounds.""",
            'summary': """Tesla's Cybertruck launched in December 2023 with a titanium body, starting at $49,990. CEO Elon Musk announced plans to produce 500,000 units annually. The truck includes autonomous driving capabilities and can tow 15,000 pounds."""
        },
        'news': {
            'source': """The Federal Reserve announced on March 20, 2024, that it would maintain interest rates at 5.25% to 5.50%. Chair Jerome Powell stated that inflation remains above the Fed's 2% target, with current rates at 3.2%. The central bank will continue monitoring economic data before making future rate decisions. Unemployment currently stands at 3.7%.""",
            'summary': """The Federal Reserve kept interest rates unchanged at 5.25% to 5.50% on March 20, 2024. Chair Jerome Powell noted inflation is at 3.2%, above the 2% target, with unemployment at 3.7%."""
        }
    }
    return jsonify(examples)

if __name__ == '__main__':
    print("=" * 70)
    print(" Hallucination Critic Web Application")
    print("=" * 70)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)