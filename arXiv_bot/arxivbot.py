import arxiv
from transformers import pipeline

# Initialize a summarization pipeline
summarizer = pipeline("summarization")

def fetch_latest_papers(topic='cs.CL', count=5):
    # Fetch the latest papers from Arxiv in the specified category (e.g., cs.CL for Computational Linguistics)
    search = arxiv.Search(
        query = f"cat:{topic}",
        max_results = count,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'summary': result.summary,
            'authors': result.authors,
            'link': result.entry_id
        })
    return papers

def summarize_papers(papers):
    # Summarize the abstracts of the papers
    summaries = []
    for paper in papers:
        # Using the summarization pipeline to summarize the text
        summarized_text = summarizer(paper['summary'], max_length=130, min_length=30, do_sample=False)
        summaries.append({
            'title': paper['title'],
            'summary': summarized_text[0]['summary_text'],
            'link': paper['link']
        })
    return summaries

# Chatbot interaction
def chatbot():
    print("Hello, I am the ArxivBot. I can fetch and summarize the newest papers from Arxiv for you.")
    
    topic = input("Please enter the topic category (e.g., cs.CL for Computational Linguistics): ").strip()
    count = int(input("How many latest papers would you like to fetch? "))
    
    print(f"Fetching the latest {count} papers from the topic {topic}...")
    papers = fetch_latest_papers(topic, count)
    
    print("I have found the following papers:")
    for paper in papers:
        print(f"- {paper['title']} (Link: {paper['link']})")

    summarize = input("Would you like a summary of these papers? (yes/no): ").strip().lower()
    if summarize == 'yes':
        print("Summarizing the papers for you...")
        summaries = summarize_papers(papers)
        for summary in summaries:
            print(f"Title: {summary['title']}")
            print(f"Summary: {summary['summary']}")
            print(f"Link: {summary['link']}\n")
    else:
        print("Okay, have a great day!")

if __name__ == "__main__":
    chatbot()

