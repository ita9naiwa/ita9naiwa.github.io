---
layout: article
titles:
  en: "Posts on Productivity and life-hack"
show_title: false
---

### Productivity boost as one of life-hacks

최근 소위 life-hack이라 하는, 생산성을 높이기 위한 방법을 연습하고 있습니다.
일할 때이건, 집에서던, 혹은 어디에서던, 어디서든 조금 더 효율적으로 일하고, 남는 시간을 조금 더 알차게 보내기 위해서입니다.
생산성 향상에 관한 포스트를 올리고, 같이 음... 활용해보면 좋을 것 같아요.

-----


### My posts on Productivity(click to read full post!)
<div class="post-list">
  <ul>
    {%- for post in site.tags.productivity -%}
    <li>
      {%- assign __path = post.url -%}
      {%- include snippets/prepend-baseurl.html -%}
      {%- assign href = __return -%}
      <div>
        <h4><a href="{{ href }}">{{ post.title }}</a></h4>
      </div>
    </li>
    {%- endfor -%}
  </ul>
</div>



<script>
  {%- include scripts/home.js -%}
</script>
