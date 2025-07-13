document.addEventListener("DOMContentLoaded", () => {
  const sendBtn = document.getElementById("send-btn");
  const input = document.getElementById("user-input");
  const chatBody = document.getElementById("chat-body");
  const darkToggle = document.getElementById("darkmode-toggle");
  const chatScreen = document.getElementById("main-chat");

  const lightBg = document.getElementById("light-bg");
  const darkBg = document.getElementById("dark-bg");

  const mobileBtn = document.getElementById("mobile-btn");
  const desktopBtn = document.getElementById("desktop-btn");

  let isSending = false;

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") { e.preventDefault(); sendMessage(); }
  });

  darkToggle.addEventListener("change", () => {
    chatScreen.classList.toggle("darkmode", darkToggle.checked);
  });

  // 📱 모바일 모드
  mobileBtn.addEventListener("click", () => {
    chatScreen.classList.remove("desktop-mode");
    chatScreen.classList.add("mobile-mode");

    lightBg.src = "Onboarding.png";
    darkBg.src  = "Onboarding2.png";
  });

  // 💻 데스크탑 모드
  desktopBtn.addEventListener("click", () => {
    chatScreen.classList.remove("mobile-mode");
    chatScreen.classList.add("desktop-mode");

    lightBg.src = "Onboarding_desktop.png";
    darkBg.src  = "Onboarding_desktop2.png";
  });

  function sendMessage() {
    if (isSending) return;
    const text = input.value.trim();
    if (!text) return;

    isSending = true;
    appendUserMessage(text);
    input.value = "";
    appendLoadingMessage();

    fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    })
    .then(res => res.json())
    .then(data => {
      removeLoadingMessage();
      appendBotMessage(data.response, data.followups ?? []);
    })
    .catch(err => {
      removeLoadingMessage();
      appendBotMessage("⚠️ 서버 연결 실패");
      console.error(err);
    })
    .finally(() => { isSending = false; });
  }

  function typeText(el, text, speed = 30, done = () => {}) {
    let i = 0;
    (function tick() {
      if (i < text.length) {
        el.append(text[i++] === "\n" ? "\n" : text[i - 1]);
        chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
        setTimeout(tick, speed);
      } else { done(); }
    })();
  }

  function appendUserMessage(msg) {
    chatBody.insertAdjacentHTML("beforeend",
      `<div class="user-message"><div class="user-bubble">${msg}</div></div>`
    );
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
  }

  function appendBotMessage(msg, followups = []) {
    const wrap = document.createElement("div");
    wrap.className = "bot-message-wrapper";

    const avatar = document.createElement("img");
    avatar.src = "images/image1.png";
    avatar.className = "bot-avatar";

    const bubble = document.createElement("div");
    bubble.className = "bot-message";

    wrap.append(avatar, bubble);
    chatBody.append(wrap);
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

    typeText(bubble, "🤖 " + msg, 30, () => {
      if (followups.length) {
        bubble.insertAdjacentHTML("beforeend",
          `<div class="followup-questions">\n\n❓ 관련 내용을 더 알고 싶다면:</div>`
        );
        const group = document.createElement("div");
        group.className = "followup-button-group";
        bubble.appendChild(group);

        followups.forEach((q, i) => {
          setTimeout(() => {
            const btn = document.createElement("button");
            btn.className = "followup-item";
            btn.textContent = q;
            btn.onclick = () => { input.value = q; sendMessage(); };
            group.appendChild(btn);
            requestAnimationFrame(() => btn.classList.add("show"));
            chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
          }, i * 300);
        });
      }
    });
  }

  function appendLoadingMessage() {
    chatBody.insertAdjacentHTML("beforeend",
      `<div class="bot-message-wrapper" id="loading">
        <img src="images/image1.png" class="bot-avatar">
        <div class="bot-message">🤖 답변을 생성 중입니다...</div>
      </div>`
    );
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
  }

  function removeLoadingMessage() {
    const el = document.getElementById("loading");
    if (el) el.remove();
  }
});