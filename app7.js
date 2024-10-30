const detect=document.querySelector(".detect");
detect.addEventListener("click",()=>{
    predictNews();
})
async function predictNews() {
    console.log("hehe");
    const newsText = document.getElementById('newsText').value;

    // Call the FastAPI endpoint using the fetch API
    const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: newsText }) // Create a JSON body with text input
    });

    // Check if response is OK
    if (response.ok) {
        const result = await response.json();
        document.getElementById('result').innerText = `Prediction: ${result.prediction}, Confidence: ${result.confidence}`;
    } else {
        document.getElementById('result').innerText = 'Error: Could not predict news';
    }
}