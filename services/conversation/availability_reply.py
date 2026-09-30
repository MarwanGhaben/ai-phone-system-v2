"""Natural, scoped no-slot replies for the restored caller flow."""


def unavailable_staff_reply(accountants, staff_id, staff_name, language):
    arabic = language == 'ar'
    name_key = 'name_ar' if arabic else 'name'
    selected = next((row for row in accountants if row.get('staff_id') == staff_id), {})
    name = selected.get(name_key) or staff_name
    alternatives = [row.get(name_key) or row.get('name') for row in accountants
                    if row.get('staff_id') and row.get('staff_id') != staff_id]
    alternatives = list(dict.fromkeys(name for name in alternatives if name))[:2]
    if arabic:
        question = ('تحب أشوف لك عند ' + ' أو '.join(alternatives) + '؟'
                    if alternatives else 'تحب أساعدك تختار وقت ثاني؟')
        return f'STAFF_UNAVAILABLE: {name} ما عنده مواعيد متاحة خلال الفترة اللي راجعتها. {question}'
    question = ('Would you like me to check ' + ' or '.join(alternatives) + '?'
                if alternatives else 'Would you like help choosing another time?')
    return f'STAFF_UNAVAILABLE: {name} has no appointments available during the period I checked. {question}'
