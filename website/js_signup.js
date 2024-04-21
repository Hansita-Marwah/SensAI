
const form = document.getElementById('loginForm');
const message = document.getElementById('message');


form.addEventListener('submit', function(event) {

event.preventDefault();


const username = document.getElementById('username').value;
const password = document.getElementById('password').value;
const email = document.getElementById('email').value;
const ph_no = document.getElementById('ph_no').value;


if (username === '' || password === '' || email === '' || ph_no === '') {
    // Display an error message
    message.textContent = 'Please fill in all fields.';
    message.style.color = '#dc3545';
} else {
    // Display a success message
    message.textContent = 'Form submitted successfully!';
    message.style.color = '#28a745';

    // Clear the form fields
    document.getElementById('username').value = '';
    document.getElementById('password').value = '';
    document.getElementById('email').value = '';
    document.getElementById('ph_no').value = '';
}
});

// Add a click event listener to the button
document.getElementById('button').addEventListener('click', function() {
  // Clear the form fields
document.getElementById('username').value = '';
document.getElementById('password').value = '';
document.getElementById('email').value = '';
document.getElementById('ph_no').value = '';
});