---
layout: article
titles:
  en: "Journals"
show_title: false
---

### 이현성의 일기


가끔 생각나는 것들을 적습니다. 주로 왜 나는 더 나은 사람이 되지 못할까...에 대해 고민할때 여기에 글을 쓰는 것 같습니다.


-----


### Posts
<div class="post-list">
  <ul>
    {%- for post in site.tags["일기"] -%}
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
