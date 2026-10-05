import pytest

from services.pt_br import to_brazilian


@pytest.mark.parametrize(
    ("portugal", "brazil"),
    [
        ("Consegues ouvir-me? O meu microfone estava a passar-se.", "Você consegue me ouvir? Meu microfone estava se passando."),
        ("Não te preocupes, não tenhas pressa.", "Não se preocupe, não tenha pressa."),
        ("Estou à espera do autocarro na estação.", "Estou esperando o ônibus na estação."),
        ("Queres tomar o pequeno-almoço comigo?", "Você quer tomar o café da manhã comigo?"),
        ("Envie-me o seu número de telefone, por favor.", "Me envie seu número de telefone, por favor."),
        ("Onde é a casa de banho?", "Onde é o banheiro?"),
        ("A minha equipa está a ganhar o jogo.", "Minha equipe está ganhando o jogo."),
        ("Não te preocupes, eu faço-o.", "Não se preocupe, eu faço."),
        ("A rapariga está a usar o telemóvel no comboio.", "A garota está usando o celular no trem."),
        ("Avisa-me se precisares de alguma coisa.", "Me avisa se você precisar de alguma coisa."),
        ("Tens a certeza? Sabes que te amo.", "Você tem certeza? Você sabe que te amo."),
        ("Tu és o melhor, dia-a-dia.", "Você é o melhor, dia-a-dia."),
    ],
)
def test_european_portuguese_becomes_brazilian(portugal: str, brazil: str) -> None:
    assert to_brazilian(portugal) == brazil


def test_brazilian_text_is_untouched() -> None:
    text = "Você tem que baixar o arquivo primeiro. Trabalhei o dia todo e estou cansado."
    assert to_brazilian(text) == text
