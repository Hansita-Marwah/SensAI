document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();


    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;

    if (username === 'username' && password === 'password') {

        showMessage('Login successful!', 'success');

    } else {

        showMessage('Invalid username or password. Please try again.', 'error');
    }
});

function showMessage(message, messageType) {
    var messageDiv = document.getElementById('message');
    messageDiv.textContent = message;

    if (messageType === 'success') {
        messageDiv.style.color = '#28a745';
    } else if (messageType === 'error') {
        messageDiv.style.color = '#dc3545';
    }


    setTimeout(function() {
        messageDiv.textContent = '';
    }, 3000);
}