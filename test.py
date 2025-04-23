class Character: 
    def __init__(self, health, power, speed):
        self.health = health
        self.power = power
        self.speed = speed
    
    def takeDamage(self, damage):
        self.health -= damage
        print(self.health)
        
class Warior(Character):
    damageModifier = 0.5
    def __init__(self, health, power, speed, wariorcast):
        super().__init__(health, power, speed)
        self.wariorcast = wariorcast
    
    def takeDamage(self, damage):
        super().takeDamage(damage*self.damageModifier)
        
character = Character(100, 50, 14)
character.takeDamage(100)
warior = Warior(100, 50, 30, "caca")
warior.takeDamage(100)
print(Warior.__dict__)