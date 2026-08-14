import pygame
import config as cfg
from ui import UI
from game import Game

pygame.init()
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()
screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
game = Game()
game_ui = UI(screen, game, None)
running = True


game_update_timer = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else:
            game_ui.handle_event(event)
    
    if game_update_timer == 6:
        if game.done:
            game.reset_game()
        else:
            game.update_game()
            game_update_timer = 0
    else:
        game_update_timer += 1

    game_ui.draw()
    pygame.display.flip()

    clock.tick(60)


pygame.quit()