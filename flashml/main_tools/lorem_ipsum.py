def lorem_ipsum(num_chars=None, num_paragraphs=None):
    """
    Generate lorem ipsum text.
    Specify EITHER `num_chars` or `num_paragraphs` (not both).
    - num_chars: int, number of characters in the output text
    - num_paragraphs: int, number of paragraphs in the output text

    Returns:
        str: Lorem ipsum text
    """
    import lorem
    if (num_chars is not None) and (num_paragraphs is not None):
        raise ValueError("Choose only one: num_chars or num_paragraphs, not both.")
    if (num_chars is None) and (num_paragraphs is None):
        raise ValueError("Specify at least one: num_chars or num_paragraphs.")

    if num_paragraphs is not None:
        # Generate the required number of paragraphs
        return "\n\n".join(lorem.paragraph() for _ in range(num_paragraphs))
    else:
        # Generate text until we have enough characters, then cut off
        result = ''
        while len(result) < num_chars:
            result += lorem.text() + ' '
        return result[:num_chars]
    
    
    


def lorem_ipsum_en(num_chars=None, num_paragraphs=None):
    from faker import Faker
    fake = Faker()
    if (num_chars is not None) and (num_paragraphs is not None):
        raise ValueError("Choose only one: num_chars or num_paragraphs, not both.")
    if (num_chars is None) and (num_paragraphs is None):
        raise ValueError("Specify at least one: num_chars or num_paragraphs.")

    if num_paragraphs is not None:
        return "\n\n".join(fake.paragraph() for _ in range(num_paragraphs))
    else:
        result = ''
        while len(result) < num_chars:
            result += fake.paragraph() + ' '
        return result[:num_chars]