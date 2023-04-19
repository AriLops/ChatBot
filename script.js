$(document).ready(function() {
    var chatbox = $('.chatbox');
    var usermsg = $('#usermsg');
    var sendbtn = $('#sendbtn');
  
    sendbtn.click(function() {
      var msg = usermsg.val();
      if (msg !== '') {
        chatbox.append('<p class="user-msg">' + msg + '</p>');
        usermsg.val('');
        $.get('/send-msg', {msg: msg}).done(function(data) {
          chatbox.append('<p class="bot-msg">' + data + '</p>');
          chatbox.scrollTop(chatbox.prop('scrollHeight'));
        });
      }
    });
  
    usermsg.keypress(function(e) {
      if (e.which == 13) {
        sendbtn.click();
      }
    });
  });