import ast
from argparse import ArgumentParser

import pandas as pd
import os

from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter, landscape


def read_csv(path):
    df = pd.read_csv(path, sep='\t')
    return df


def clusterRead(fname):
    words = []
    words_idx = []
    cluster_idx = []
    sent_idx = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            parts = line.split("|||")

            cluster_id = int(parts[4])
            word_id = int(parts[3])
            sent_id = int(parts[2])

            cluster_idx.append(cluster_id)
            words_idx.append(word_id)
            sent_idx.append(sent_id)
            words.append(parts[0])

    return words, words_idx, sent_idx, cluster_idx


def get_concept_map(words, cluster_idx):
    cloudMap = {}
    prev = cluster_idx[0]
    thisCloud = []

    for i, w in enumerate(cluster_idx):

        if (prev == cluster_idx[i]):
            thisCloud.append(words[i])
        else:
            cloudMap[prev] = thisCloud
            thisCloud = []
            prev = cluster_idx[i]

    cloudMap[prev] = thisCloud
    return cloudMap


def createWordCloud(wordList, name):
    # Convert the list of words into a space-separated string
    text = ' '.join(wordList)

    # Create a WordCloud object
    stopwords=set() # Adding Part
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)

    # check if the word cloud folder exists
    if not os.path.exists("word_cloud_images"):
        os.makedirs("word_cloud_images")

    # generate a image of word cloud
    wordcloud.to_file("word_cloud_images/"+name+".png")

    # Display the generated word cloud using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')  # Turn off axis numbering
    # plt.show()


def save_all_concept_sentences(concept_sentence_df, concept_map):
    text = []
    image_filename = []

    # sort dataframe by the value of concept
    concept_sentence_df = concept_sentence_df.sort_values(by='concept')
    gender_related_concepts = concept_sentence_df['concept'].tolist()
    stat = concept_sentence_df['statistic'].tolist()

    for i, value in enumerate(gender_related_concepts):
        concept_words = concept_map[value]
        createWordCloud(concept_words, "concept_"+str(value))
        image_filename.append("word_cloud_images/concept_"+str(value)+".png")

        concept_stat = ast.literal_eval(stat[i])
        female_percentage = concept_stat[0]
        male_percentage = concept_stat[1]
        others_percentage = concept_stat[2]
        total_count = concept_stat[3]


        sentences = ("\nPercentage of sentences related to this concept: " + str(round(total_count/1631*100, 2)) + "%" +
                     "\nFemale related Prediction (%): " + str(female_percentage) + "%" +
                     "\nMale related Prediction (%): " + str(male_percentage) + "%" +
                     "\nOthers Prediction (%): " + str(others_percentage) + "%\n"  +
                     "\nAll sentences with the concept:\n")


        # convert string to list
        sentence_list = ast.literal_eval(concept_sentence_df[concept_sentence_df['concept'] == value]['sentences'].tolist()[0])
        for sentence in sentence_list:
            sentences = sentences + sentence + "\n"
        text.append(sentences)
    return text, image_filename


def create_pdf(output_filename, image_name, text):
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(letter))

    # 创建一个流，其中我们将放置报告的元素
    story = []

    for i in range(len(image_name)):
        styles = getSampleStyleSheet()
        content = image_name[i].split("/")[1].split(".")[0].replace("_", " ")
        style = styles["Normal"]
        style.fontName = 'Helvetica-Bold'  # 设置为加粗
        style.underline = True  # 设置下划线
        txt_content = Paragraph(content, style)
        story.append(txt_content)

        # 添加图片
        img = Image(image_name[i], 6*inch, 3*inch)
        story.append(img)

        # 添加文本
        styles = getSampleStyleSheet()
        text[i] = text[i].replace("\n", "<br/>")
        txt_content = Paragraph(text[i], styles["Normal"])
        story.append(txt_content)

        # 添加分节符，将内容移动到下一页
        story.append(PageBreak())

    # 创建PDF
    doc.build(story)


def main():
    parser = ArgumentParser()
    parser.add_argument("--concept_sentence_file", type=str, required=True)
    parser.add_argument("--concept_cluster_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    argparse = parser.parse_args()

    concept_sentence_df = read_csv(argparse.concept_sentence_file)
    words, words_idx, sent_idx, cluster_idx = clusterRead(argparse.concept_cluster_file)
    concept_map = get_concept_map(words, cluster_idx)

    text, image_filename = save_all_concept_sentences(concept_sentence_df, concept_map)
    create_pdf(argparse.output_file, image_filename, text)


if __name__ == "__main__":
    main()



