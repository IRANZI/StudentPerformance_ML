document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {
        hours_studied: parseFloat(formData.get('hours_studied')),
        previous_scores: parseFloat(formData.get('previous_scores')),
        extracurricular: formData.get('extracurricular'),
        sleep_hours: parseFloat(formData.get('sleep_hours')),
        sample_papers: parseFloat(formData.get('sample_papers'))
    };

    if (data.hours_studied + data.sleep_hours > 24) {
        alert('Total hours studied and sleep hours cannot exceed 24!');
        return;
    }

    const btn = e.target.querySelector('button');
    const originalText = btn.innerText;
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch('/performance/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok) {
            document.getElementById('score').innerText = result.predicted_performance_index;
            document.getElementById('fairness-msg').innerText = result.fairness_statement;
            document.getElementById('result-container').classList.remove('hidden');
            
            // Smooth scroll to results
            document.getElementById('result-container').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            alert('Error: ' + (result.error || 'Something went wrong'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the server.');
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
