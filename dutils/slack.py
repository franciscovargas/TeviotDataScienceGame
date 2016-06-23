import slacker


def send_results(msg):
    token = open("../token").read().strip("\n")
    slack = slacker.Slacker(token)
    slack.chat.post_message('#automated_results',
                            'results: \n %s' % msg)


if __name__ == "__main__":
    send_results(" Wooohoo python NN chatbot working ")


