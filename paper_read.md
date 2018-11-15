---
layout: article
titles:
  en: "Review Papers"
show_title: false
---

#### Note:
논문을 가끔 읽거나, 책을 읽고 공부하거나, 한 것들을 정리합니다.

#### Posts on Read Papers
<div class="post-list">
  <ul>
    {%- for post in site.tags["study"]-%}
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
