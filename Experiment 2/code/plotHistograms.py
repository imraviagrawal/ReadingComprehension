import train
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_set = train.load_train_dataset("data/squad")
    questions, contexts, answers = zip(*training_set)
    question_lengths = map(len, questions)
    context_lengths = map(len, contexts)
    answer_lengths = [a[1] - a[0] + 1 for a in answers]
    plt.figure(1)
    plt.xlabel('Question Lengths')
    plt.ylabel('Counts')
    plt.hist(question_lengths, 60)
    plt.figure(2)
    plt.xlabel('Context Paragraph Lengths')
    plt.ylabel('Counts')
    plt.hist(context_lengths, 200)
    plt.figure(3)
    plt.xlabel('Answer Lengths')
    plt.ylabel('Counts')
    plt.hist(answer_lengths, 60)
    plt.show()
