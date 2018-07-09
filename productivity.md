---
layout: article
titles:
  en: "Posts on Productivity and life-hack"
show_title: false
---

### Productivity boost as one of life-hacks

Recently I'm practicing some life-hacks to boost my performance(in works, house, or anywhere).
I wrote this post in either Korean, or English to share someone else.

Posts on productivitiy helps at least two points.

1. I'm obliged to write/practice something practical.
2. hmm... not well.


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
