---
#
# Use the widgets beneath and the content will be
# inserted automagically in the webpage. To make
# this work, you have to use › layout: frontpage
#
layout: frontpage
header:
  image_fullwidth: header_unsplash_12.jpg
widget1:
  title: "Kim"
  url: 'http://sml09181.github.io/ai/projects'
  image: 호빵.png
  text: '왤케 <em>졸리지</em> 호빵 먹고 싶다.'
widget2:
  title: "Su"
  url: 'http://sml09181.github.io/cs/projects'
  text: '<em>기울임체</em> 신기하다.<br/>1. 인공지능 재밌다<br/>2. Sujin&#39;s Ocean.<br/>3. 깃허브 주소는 <a href="http://github.com/sml09181/">Go to Github</a>.<br/>4. 아샷추 먹고 싶다...<br/>5. 블록체인 너무 매력적'
  video: '<a href="#" data-reveal-id="videoModal"><img src="http://sml09181.github.io/images/노르웨이.jpg" width="302" height="182" alt=""/></a>'
widget3:
  title: "Jin"
  url: 'http://sml09181.github.io/blockchain/projects'
  image: 블록체인.jpg
  text: '이거 공개 비공개 기능 없나'
#
# Use the call for action to show a button on the frontpage
#
# To make internal links, just use a permalink like this
# url: /getting-started/
#
# To style the button in different colors, use no value
# to use the main color or success, alert or secondary.
# To change colors see sass/_01_settings_colors.scss
#
callforaction:
  url: https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox?compose=new
  text: Contact Me by Google Email ›
  style: mainbutton
permalink: 
#
# This is a nasty hack to make the navigation highlight
# this page as active in the topbar navigation
#
homepage: true
---

<div id="videoModal" class="reveal-modal large" data-reveal="">
  <div class="flex-video widescreen vimeo" style="display: block;">
    <iframe width="1280" height="720" src="https://www.youtube.com/embed/3b5zCFSmVvU" frameborder="0" allowfullscreen></iframe>
  </div>
  <a class="close-reveal-modal">&#215;</a>
</div>
