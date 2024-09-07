const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const toggler = document.querySelector(".chatbot-toggler");

sendButton.addEventListener('click', () => {
    const messageText = messageInput.value.trim();

    if (messageText !== '') {
        const inputMessage = createMessage(messageText, 'input-message');
        chatMessages.appendChild(inputMessage);
        // 发送用户输入到后端
        fetch('/process_input', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: messageText }),
        })
        .then(response => response.json())
        .then(data => {
            const outputMessage = createMessage(data.output, 'output-message');
            chatMessages.appendChild(outputMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });

        messageInput.value = '';
    }
});

function createMessage(text, className) {
    const Container = document.createElement('div');
    if (className === 'input-message') {

        Container.style.display = "flex";
        Container.style.alignSelf = "flex-end";
        Container.style.flexDirection = "row";
        Container.style.marginRight = "0";

        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', className);
        const inputColumn = document.createElement('div');
        inputColumn.style.display = "flex";
        inputColumn.style.alignSelf = "flex-end";
        inputColumn.style.maxWidth = "75%";
        inputColumn.style.flexDirection = "column";
        const nameDiv = document.createElement('div');
        nameDiv.textContent = "你";
        nameDiv.style.color = "black";
        nameDiv.style.marginRight = "8px";
        nameDiv.style.alignSelf = "flex-end";
        nameDiv.classList.add('input-name');
        inputColumn.appendChild(nameDiv);

        const textDiv = document.createElement('div');
        textDiv.textContent = text;
        textDiv.classList.add('text');
        messageContainer.appendChild(textDiv);
        inputColumn.appendChild(messageContainer);


        const avatarImg = document.createElement('img');
        avatarImg.src = "static/image/avatar.jpg";
        avatarImg.alt = "机器人";
        avatarImg.style.width = "60px";
        avatarImg.style.marginLeft = "8px";
        avatarImg.style.justifySelf = 
        avatarImg.style.marginTop = "-15px";
        avatarImg.style.marginRight = "0";


        Container.appendChild(inputColumn);
        Container.appendChild(avatarImg);

    }

    if (className === 'output-message'){
        Container.style.display = "flex";
        Container.style.flexDirection = "row";
        Container.style.alignSelf = "flex-start";
        Container.style.margin = "0";
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', className);
        const inputColumn = document.createElement('div');
        inputColumn.style.display = "inline-block";
        inputColumn.style.alignSelf = "flex-start";
        inputColumn.style.maxWidth = "75%";
        inputColumn.style.display = "flex";
        inputColumn.style.flexDirection = "column";
        const nameDiv = document.createElement('div');
        nameDiv.textContent = "云游智灵";
        nameDiv.style.color = "black";
        nameDiv.style.marginLeft = "8px";
        nameDiv.style.alignSelf = "flex-start";
        nameDiv.classList.add('input-name');
        inputColumn.appendChild(nameDiv);

        const textDiv = document.createElement('div');
        textDiv.textContent = text;
        textDiv.classList.add('text');
        messageContainer.appendChild(textDiv);
        inputColumn.appendChild(messageContainer);

        const avatarImg = document.createElement('img');
        avatarImg.src = "static/picture/globe-solid.svg";
        avatarImg.alt = "机器人";
        avatarImg.style.width = "30px";
        avatarImg.style.marginRight = "8px";
        avatarImg.style.marginTop = "-45px";
        Container.appendChild(avatarImg);
        Container.appendChild(inputColumn);
    }



    return Container;
}





