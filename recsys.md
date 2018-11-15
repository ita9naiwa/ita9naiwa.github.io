---
layout: article
titles:
  en: "Recommender Systems"
show_title: false
---

### My posts on Recommender Systems(click to read full post!)
<div class="post-list">
  <ul>
    {%- for post in site.tags["recommender systems"]-%}
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
