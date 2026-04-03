document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('analyze-btn');
    const input = document.getElementById('news-input');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');
    
    const resultCard = document.getElementById('result-card');
    const errorMsg = document.getElementById('error-msg');
    
    const sentimentIcon = document.getElementById('sentiment-icon');
    const predictionText = document.getElementById('prediction-text');
    const confidenceScore = document.getElementById('confidence-score');
    const progressBar = document.getElementById('progress-bar');
    
    btn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) return;
        
        // Setup UI for loading
        btn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        resultCard.classList.add('hidden');
        errorMsg.classList.add('hidden');
        
        // Reset classes
        resultCard.className = 'result-card glassmorphism hidden';
        progressBar.style.width = '0%';
        
        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            
            const data = await resp.json();
            
            if (!resp.ok) {
                throw new Error(data.error || 'Server error');
            }
            
            // Populate success UI
            showResult(data);
        } catch (err) {
            errorMsg.textContent = err.message;
            errorMsg.classList.remove('hidden');
        } finally {
            btn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    });
    
    function showResult(data) {
        const pLabel = data.prediction.toLowerCase();
        const score = (data.confidence * 100).toFixed(1);
        
        predictionText.textContent = data.prediction;
        confidenceScore.textContent = `${score}%`;
        
        if (pLabel === 'positive') {
            sentimentIcon.textContent = '📈';
            resultCard.classList.add('pos-theme');
        } else if (pLabel === 'negative') {
            sentimentIcon.textContent = '📉';
            resultCard.classList.add('neg-theme');
        } else {
            sentimentIcon.textContent = '➖';
            resultCard.classList.add('neu-theme');
        }
        
        resultCard.classList.remove('hidden');
        
        // Animate progress bar slightly after render
        setTimeout(() => {
            progressBar.style.width = `${score}%`;
        }, 50);
    }
});
