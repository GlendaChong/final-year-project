for i in {1..41}; do
  curl -H "Authorization: Bearer your_jina_api_key" \
       -o round2/techxplore_pages/page${i}.html \
       "https://r.jina.ai/https://techxplore.com/computer-sciences-news/page${i}.html"
  echo "Saved page ${i}"
  sleep 2
done

# replace your_jina_api_key with your actual Jina API key before running the script.
