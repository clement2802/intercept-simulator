import itertools
import pygame
import time
# from ipdb import set_trace as st
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import numpy as np
import pyglet
from gym import core, spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding
from scipy.spatial import distance_matrix
from ray.tune.registry import register_env

import pandas as pd
import matplotlib.pyplot as plt

# Spécificités du simulateur (habillage)
NB_VAISSEAUX    = 1
NB_ASTEROIDES   = NB_VAISSEAUX
NB_ACTIONS_POSSIBLES_PAR_VAISSEAU = 2
NB_OBSERVATION_UNITAIRE = 3
VITESSE_VAISSEAUX = 2
VITESSE_VAISSEAUX_Y_MAX = 8
OFFSCREEN_SPACE = 1
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Affect'a Meute"
LEFT_LIMIT = OFFSCREEN_SPACE
RIGHT_LIMIT = SCREEN_WIDTH - OFFSCREEN_SPACE
BOTTOM_LIMIT = OFFSCREEN_SPACE # limite haute
TOP_LIMIT = SCREEN_HEIGHT - 60 # limite du bas de l'écran
capacité_res = 1000

#Représentation des astéroides
DISTANCE_COLLISION = 20 # distance à laquelle on considère qu'il y a eu collision
RAYON_ASTEROIDES = 10
COULEUR_ASTEROIDES_R = 200
COULEUR_ASTEROIDES_G = 0.8
COULEUR_ASTEROIDES_B = 0

#Dimensions et couleurs des vaisseaux
LONGUEUR_VAISSEAU = 20
LARGEUR_VAISSEAU = 5
red = np.random.randn(NB_VAISSEAUX)
green = np.random.randn(NB_VAISSEAUX)
blue = np.random.randn(NB_VAISSEAUX)

class Asteroide():
    """ Représentation d'un asteroide """

    def __init__(self, id_asteroide):
        """ Positionnement initial de l'astéroide decpuis la droite avec un
        déplacement de droite à gauche """
        self.id         = id_asteroide
        self.speed      = VITESSE_VAISSEAUX
        # self.center_x   = RIGHT_LIMIT - np.random.choice([0,1,2,3,4,5,6])*100
        self.center_x   = RIGHT_LIMIT
        # self.center_x   = RIGHT_LIMIT
        self.center_y   = np.random.randint(SCREEN_HEIGHT)
        self.change_x   = - self.speed
        self.change_y   = 0

    def update(self):
        """ Mise à jour dde la position du vaisseau """

        # propagation de dynamique
        self.center_x += self.change_x

        # Si l'astéroide sort de l'écran
        if self.center_x > RIGHT_LIMIT:
            self.center_x = LEFT_LIMIT
        if self.center_x < LEFT_LIMIT:
            self.center_x = RIGHT_LIMIT


class Vaisseau():
    """ Représentation d'un vaisseau """
    def __init__(self, id_vaisseau):
        """ Set up the space ship. """
        self.id         = id_vaisseau
        self.change_dir = 0
        self.speed      = VITESSE_VAISSEAUX
        self.max_speed  = VITESSE_VAISSEAUX
        self.center_x   = LEFT_LIMIT
        self.center_y   = np.random.randint(SCREEN_HEIGHT)
        self.change_x   = self.speed
        self.change_y   = 0
        self.angle      = 0 # Pointe vers la droite
        self.inactif    = False
        self.reservoir  = capacité_res
        self.isEmpty = False

    def act(self, action, informer = False):
        """ Réalisation d'un ordre commandé """

        if self.reservoir > 0:
            if action == 0:
                # Déplacement en crabe
                self.change_dir = 1
                self.reservoir -= 1
                # if informer:
                #     print("\tVaisseau", self.id, " --> Boost latéral +")

            elif action == 1:
                self.change_dir = -1
                self.reservoir -= 1
                # if informer:
                #     print("\tVaisseau", self.id, " --> Boost latéral -")


    def update(self):
        """ Mise à jour de la position du vaisseau """

        # propagation de dynamique
        self.center_x += self.change_x
        self.center_y += self.change_y

        # Application d'un DV latéral plafonné
        INCREMENT_VITESSE_LATERAL = 0.5
        if self.change_dir == 1:
            new_change_y = self.change_y + INCREMENT_VITESSE_LATERAL
            if new_change_y <= VITESSE_VAISSEAUX_Y_MAX:
                self.change_y = new_change_y
            else:
                self.change_y = VITESSE_VAISSEAUX_Y_MAX
        elif self.change_dir == -1:
            new_change_y = self.change_y - INCREMENT_VITESSE_LATERAL
            if new_change_y >= -VITESSE_VAISSEAUX_Y_MAX:
                self.change_y = new_change_y
            else:
                self.change_y = VITESSE_VAISSEAUX_Y_MAX

        # If the ship goes off-screen, move it to the other side of the window
        if self.center_y >= TOP_LIMIT:
            self.center_y = TOP_LIMIT
            self.change_y -= INCREMENT_VITESSE_LATERAL

        if self.center_y <= BOTTOM_LIMIT:
            self.center_y = BOTTOM_LIMIT
            self.change_y += INCREMENT_VITESSE_LATERAL

        #Si le vaisseau sort de l'écran par la droite, il est hors jeu
        if self.center_x > RIGHT_LIMIT:
            self.inactif = True


class AffectaMeuteEnv(core.Env):
    """
    Des vaisseaux venant de la gauche doivent intercepter des astéroides
    provenant de la droite par contact direct

    Spécificités:
        Deplacement uniquement en crabe
        Orientation du vaisseau autour de son CdG
        Vitesse d'avance constante
        Mouvement relatif de corps selon l'axe X de l'écran
        Pas de spliting lors de la destruction d'un astéroide

    Objectif:
        Entrainer un PPO/DQN via la RLLIB
        Trouver les hyperparamètre via Tune de RLlib

    **STATE:**
    L'état est composé pour tous les couples vaisseau-astéroide de:
      - miss-distance à l'astéroide (distance suivant Y à l'astéroide)
      - Vy, vitesse relative entre asteroide et vaisseau suivant l'axe Y

    **ACTIONS:**

    Chaque vaisseau peut soit:
        - Ne rien faire
        - Faire un boost vers le haut
        - Faire un boost vers le bas

        Dans une future version avec visibilité on aura:
        - Tourner autour de son CdG en sens trigonométrique
        - Tourner autour de son CdG en sens horaire

    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    def __init__(self):

        # L'état est en position relative:
        #   - distance à l'astéroide le plus proche (norme de la LOS)
        #   - angle entre l'axe vaisseau et la LOS (0 quand on pointe sur l'astéroide)
        #   - Vx, vitesse relative entre asteroide et vaisseau suivant l'axe x
        #   - Vy, vitesse relative entre asteroide et vaisseau suivant l'axe y
        nb_couple_VaisAst = len([x for x in itertools.product(range(NB_VAISSEAUX), range(NB_ASTEROIDES))]) 
        self.dim_vobs = NB_OBSERVATION_UNITAIRE * nb_couple_VaisAst + NB_VAISSEAUX + 1 # on rajoute un reservoir par vaisseau présent
        lemax = np.linalg.norm([SCREEN_WIDTH,SCREEN_HEIGHT]) # sqrt(800*800 + 600*600)
        self.observation_space = spaces.Box(low=-lemax, high=lemax, shape=(self.dim_vobs,), dtype=np.float32)


        self.act = 10 # parametre qui permet à la fonction de rendering de connaitre l'action
        self.id_ship_act = -5 # parametre qui donne l'id_ship du vaisseau qui reçoit l'action actuelle

        self.misDist = np.zeros((NB_VAISSEAUX, NB_ASTEROIDES))
        self.previous_miss_dist = np.zeros((NB_VAISSEAUX, NB_ASTEROIDES))
        self.iteration = 0

        #L'IA peut soit:
        #    - Ne rien faire
        # soit maneuvrer un vaisseau:
        #   - Faire un boost vers le haut
        #   - Faire un boost vers le bas
        #   - (ultérieurement) Tourner autour de son CdG en sens trigonométrique
        #   - (ultérieurement) Tourner autour de son CdG en sens horaire
        #
        # Exemple pour un scénario avec 2 vaisseaux:
        #  [vaisseau 1 boost+, vaisseau 1 boost-, vaisseau 2 boost+, vaisseau 2 boost-, Ne rien faire ]

        self.action_space = spaces.Discrete(NB_VAISSEAUX * NB_ACTIONS_POSSIBLES_PAR_VAISSEAU + 1)

        # Spécificités Gym
        self.viewer = None
        self.state = None
        self.seed()
 
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_status_vaisseaux(self):
        """ Mise à jour de l'état de participation des vaisseaux à la meute """
        self.ships_alive = [not ship.inactif for ship in self.ships]
        # si le vaisseau atteint ou dépasse l'ast il devient inactif
    def reset(self):
        """ Set up the game and initialize the variables. """
        self.score = 0
        self.frame_count = 0
        self.game_over = False

        # Initialisation des vaisseaux
        # Les vaisseaux sont créés et considérés actifs.
        # Si un vaisseau subit une collision, il n'est pas retiré de
        # la simulation mais rendu INACTIF. Ceci permet de gérer les
        # indices du vecteur d'actions possibles
        self.ships = [Vaisseau(id_vaisseau) for id_vaisseau in range(NB_VAISSEAUX)]
        self.update_status_vaisseaux()

        # Initialisation des astéroides
        self.asteroides = [Asteroide(id_asteroide) for id_asteroide in range(NB_ASTEROIDES)]


        # Répartition des asteroides sur toute la hauteur de l'écran
        #posY_ast = np.random.uniform(low=0.0, high=SCREEN_HEIGHT, size=NB_ASTEROIDES) # Induit que dans certaines configuration deux asteroides peuvent etre proches et on peu alors les détruires avec un seul vaisseau
        posY_ast = np.linspace(start=250, stop=SCREEN_HEIGHT-250, num=NB_ASTEROIDES)
        # for i_ast in range(len(self.asteroides)):
        #     self.asteroides[i_ast].center_y = TOP_LIMIT - np.random.choice([2,3,4,5,6,7,8,9,10])*50
        for i_ast in range(len(self.asteroides)):
            self.asteroides[i_ast].center_y = posY_ast[i_ast]


        # Retourne l'état courant
        self.state = self.observation()

        #
        # print(f"dim_vobs = {self.dim_vobs}")
        # print("espace d'observations : ",self.observation_space)
        # print("espace d'actions : ",self.action_space)


        return self.state

    def step(self, action, informer = True):
        """ Evolution de l'environnement sur un pas """

        ######################################################
        # Affichage pour l'utilisateur de l'état courant et de
        #  l'action selectionnée par l'IA
        ######################################################
        if informer:
            # -- Status sur les menaces
            # print("\n", "-"*50, "\nMenace(s) : ", NB_ASTEROIDES, " astéroïde(s) dans le scénario :")
            # for asteroide in self.asteroides:
            #     print("\tAsteroide", asteroide.id, " Toujours menaçant")
            # # -- Status sur les Vaisseaux
            # print("Vaisseau(x):")
            # for idship in range(len(self.ships)):
            #     ship = self.ships[idship]
            #     if ship.inactif:
            #         print("\tShip", idship, " Dead")
            #     else:
            #         print("\tShip", idship, " Actif")
            # -- Observation courante
            print("Observation:")
            i_info = 0
            for i_ship in range(NB_VAISSEAUX):
                for i_ast in range(NB_ASTEROIDES):
                    print("\tShip", i_ship, " Ast", i_ast, ": MDis Y:", self.state[i_info])
                    i_info += 1
                    print("\tShip", i_ship, " Ast", i_ast, ": Vrel Y:", self.state[i_info])
                    i_info += 1
                    print("\tShip", i_ship, " Ast", i_ast, ": XDist :", self.state[i_info])
                    i_info += 1
            for i_ship in range(NB_VAISSEAUX):
                print("\tCarburant ship ", i_ship, " : ", self.state[i_info])
                i_info += 1

            print("\tVitesse relative commune en x : ", self.state[i_info])

            # -- Action selectionnée
            # print("Action:", self.act)


        # Réalisation de l'action. On fait le lien entre l'ordre donné par l'IA
        # et l'action d'un vaisseau
        #
        #  1) On souhaite que le vecteur des actions possibles reste constant
        #       pour toute une simulation. C'est-à-dire que si le vaisseau 2 a
        #       été détruit (action=2 et action=3), il faut que ces actions ne
        #       menent à rien ensuite et qu'elles ne commandent pas le
        #       vaisseau 3 (action=4 et action=5)
        #
        #  2) L'action unique d'indice "action_space.n-1" = NE RIEN FAIRE

        # L'action demandée est-elle == Ne rien faire ?
        if action == self.action_space.n-1:
            self.act = action

        elif action != (self.action_space.n-1):

            # Détermination du vaisseau concerné par l'action
            id_ship = int(np.floor(action/NB_ACTIONS_POSSIBLES_PAR_VAISSEAU))
            #print("\nAction: ", action, " pour vaisseau id_ship: ", id_ship)
            self.id_ship_act = id_ship

            # Si l'action porte sur un vaisseau inactif, on ne fait rien.
            # Sinon on agit
            if self.ships_alive[id_ship]:

                # Détermination de l'action
                action_ship = action % NB_ACTIONS_POSSIBLES_PAR_VAISSEAU
                self.act = action_ship

                # Réalisation de l'action
                self.ships[id_ship].act(action_ship, informer)

        # else:
        #     if informer:
        #         print("\tNE RIEN FAIRE")

        # On laisse l'environnement réagir
        nouvel_etat, reward, done = self.update()

        # if informer:
        #     # -- Reward immédiat
        #     print("Reward:", reward)

        return (nouvel_etat, reward, done, {})

    def random_rollout(self): # A tester !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """ A rollout is a simulation of a policy in an environment.
         The code below performs a random rollout. It takes random
         actions until the simulation has finished and returns the
         cumulative reward.
        """

        state = self.reset()

        done = False
        cumulative_reward = 0

        # Keep looping as long as the simulation has not finished.
        while not done:
            # Choose a random action
            action = np.random.choice(self.action_space)

            # Take the action in the environment.
            state, reward, done, _ = self.step(action)

            # Update the cumulative reward.
            cumulative_reward += reward

        # Return the cumulative reward.
        return cumulative_reward

    def update(self):
        """ Mise à jour de tout l'environnement avec:
                Propagation de la dynamique
                Détermination des collisions
        """
        # print("itération ", self.iteration)
        self.iteration += 1

        fin_execution = False
        reward = 0

        # --- Propagation des états ---
        if (len(self.asteroides) > 0) and (len(self.ships) > 0):
            [asteroide.update() for asteroide in self.asteroides]
            [ship.update() for ship in self.ships]
            self.update_status_vaisseaux()

        # reward distribué de manière continue, basé sur la matrice des valeurs absolues des miss distances
        # md = np.zeros(shape=(NB_VAISSEAUX, NB_ASTEROIDES))
        # for ship in self.ships:
        #     for ast in self.asteroides:
        #         md[ship.id][ast.id] = ast.center_y - ship.center_y
        #         md[ship.id][ast.id] = np.interp(md[ship.id][ast.id], [-SCREEN_HEIGHT, SCREEN_HEIGHT], [-1, 1])
        #         if ship.inactif == False and np.abs(md[ship.id][ast.id]) <= np.abs(np.interp(DISTANCE_COLLISION, [-SCREEN_HEIGHT, SCREEN_HEIGHT], [-1, 1])) :
        #             reward += 2
        # self.misDist = md
        # print("mistDist : ",self.misDist)

        #
        # if self.act != 4 and self.ships[self.id_ship_act].reservoir != 0:
        #     reward -= 1

        # --- Gestion des collisions entre vaisseaux et astéroides ---
        if np.any(self.ships_alive) and (len(self.asteroides) > 0):

            # Calcul des distances entre astéroides et vaisseaux encore actifs
            XY_vaisseaux  = [(ship.center_x, ship.center_y) for ship in self.ships if ship.inactif == False]
            XY_asteroides = [(ast.center_x, ast.center_y) for ast in self.asteroides]
            dist = distance_matrix(XY_vaisseaux, XY_asteroides) # donne dans une matrice les distances entre chaque vaisseau et astéroïde à la fin de l'itération

            # Récupération des couples (vaisseau actif, asteroide) ayant une
            # distance relative d'interception
            id_ship, id_ast = np.where(dist < DISTANCE_COLLISION)
            # print(f"\nid_ship = {id_ship}")
            # print(f"id_ast = {id_ast}")
            # print(f"reward = np.unique(id_ast) = {np.unique(id_ast)}")

            if len(id_ship) > 0:
                # !!!!! -------------- !!!!!
                #   On kill un astéroide
                #        WIN +1
                # !!!!! -------------- !!!!!
                # Cette implémenation est robuste à plusieurs vaisseaux qui
                # impactent le meme asteroide. On détruit plusieurs vaisseaux
                # mais un seul astéroide
                reward += len(id_ship) # nombre de collision qui ont eu lieu

                # On flag comme inactif le vaisseau qui a été détruit
                # Le rendre inactif supprime sa participation dans la meute
                vaisseaux_actifs = [ship for ship in self.ships if ship.inactif == False]
                ship_a_retirer = [vaisseaux_actifs[i] for i in np.unique(id_ship) ]
                for ship_devenu_inactif in ship_a_retirer:
                    # Désactivation du vaisseau
                    ship_devenu_inactif.inactif = True

                # Mise à jour des vaisseaux actifs
                self.update_status_vaisseaux()

                # Retrait de l'astéroide détruit
                ast_a_retirer = [self.asteroides[i] for i in np.unique(id_ast)]
                for ast_retrait in ast_a_retirer:
                    # Suppression de l'astéroide
                    self.asteroides.remove(ast_retrait)

        # --- Gestion des vaisseaux inactifs ---
        # Un vaisseau devient inactif soit :
        # 1) Par destruction (cf. ci-dessus)
        # 2) Parce qu'il est sorti de l'écran par la droite (cf. Vaisseau.update)
        # 3) Parce qu'il a dépassé toutes les menaces (ce qui est fait dans les lignes suivantes)
        pos_X_asteroides = np.array([ast.center_x for ast in self.asteroides])
        # pos_X_vaisseaux = np.array([ship.center_x for ship in self.ships])

        for ship in self.ships:
            if np.all(ship.center_x >= pos_X_asteroides):
                # Suppression du vaisseau de la meute. Mais pas de point de pénalité
                ship.inactif = True
        for ship in self.ships:
                if ship.inactif == True:
                    fin_execution = True

        # --- Gestion de la fin de simulation ---
        # s = 0
        # for ship in self.ships:
        #     if ship.inactif == False:
        #         s += ship.reservoir
        if (len(self.asteroides) == 0) or (not np.any(self.ships_alive)):
            # Si tous les Astéroides ont été détruits, on a gagné
            # Si tous les vaisseaux sont inactifs, on a terminé
            fin_execution = True

        if fin_execution:
            reward -= len(self.asteroides)

        # Acquisition de l'état
        self.state = self.observation()

        # Mise à jour du score pour l'affichage
        self.score += reward

        return (self.state, reward, fin_execution)

    def observation(self):
        """ Création d'une observation

            L'état est composé pour tous les couples vaisseau-astéroide de:
                - miss-distance à l'astéroide (distance suivant Y à l'astéroide)
                - Vy, vitesse relative entre asteroide et vaisseau suivant l'axe Y

             Afin de gérer au mieux la disparition de vaisseau ou d'astéroides,
             On crée un vecteur d'observation de taille fixe:
              - Toutes les observations relatives à un vaisseau inactif sont mises à 0
              - Toutes les observations où intervient un astéroide détruit sont mise à 0
        """

        # La taille de l'observation est constante pendant toute la simulation.
        # On gère le padding en definissant une observation vide et remplissant
        # les éléments existants
        obs = np.zeros(self.dim_vobs, dtype=np.float32)

        #print("Vaisseau actifs", self.ships_alive)
        #print("Astéroides existants:", [asteroide.id for asteroide in self.asteroides])

        index = 0
        # Condition de traitement
        if (len(self.ships) != 0) and (len(self.asteroides) != 0):

            # Réalisation du traitement unitaire pour chaque vaisseau

            for id_ship in range(len(self.ships)):
                ship = self.ships[id_ship]

                # Si le vaisseau est inactif, on laisse toutes ses
                # observations à 0 (il ne participe plus dans la meute)
                if not ship.inactif:
                    # Mise à jour des observations par rapport à chaque menace
                    for men in self.asteroides:

                        # INFO 1: Distance latérale à l'astéroide (miss-distance)
                        ligne = id_ship*NB_OBSERVATION_UNITAIRE*NB_ASTEROIDES+ men.id*NB_OBSERVATION_UNITAIRE
                        # print("ligne:", ligne)
                        obs[ligne] = men.center_y - ship.center_y

                        # normalisation par rapport à la hauteur de l'ecran
                        # obs[ligne] = np.interp(obs[ligne], [-SCREEN_HEIGHT, SCREEN_HEIGHT], [-1, 1])
                        # normalisation par rapport à la hauteur de l'ecran
                        # obs[ligne] = np.interp(obs[ligne], [-SCREEN_HEIGHT, SCREEN_HEIGHT], [-1, 1])

                        # INFO 2: Vitesse latérale de rapprochement à l'astéroide
                        ligne += 1
                        #print("ligne:", ligne)
                        obs[ligne] = men.change_y - ship.change_y
                        # obs[ligne] = np.interp(obs[ligne], [-VITESSE_VAISSEAUX_Y_MAX, VITESSE_VAISSEAUX_Y_MAX], [-1, 1])

                        # INFO 3: Distance en x à l'asteroide
                        ligne += 1
                        # print("ligne:", ligne)
                        obs[ligne] = men.center_x - ship.center_x
                        # obs[ligne] = np.interp(obs[ligne], [LEFT_LIMIT, RIGHT_LIMIT], [-1, 1]) if np.interp(obs[ligne], [LEFT_LIMIT, RIGHT_LIMIT], [-1, 1]) > 0 else 0
                        ligne += 1
                    index = ligne

                    # Niveau de carburant de chaque vaisseau
                    obs[index + id_ship] = 0
            obs[index+NB_VAISSEAUX] = men.speed - ship.speed

                # else:
                #     # On fixe des bservations très grandes si le vaisseau est inactif
                #     # L'objectif étant de minimiser les miss-distances, et les dist_x, les fixer a 0 a la fin de l'execution fausse les resultats
                #     for men in self.asteroides:
                #
                #         # INFO 1: Distance latérale à l'astéroide (miss-distance)
                #         ligne = id_ship*NB_OBSERVATION_UNITAIRE*NB_ASTEROIDES+ men.id*NB_OBSERVATION_UNITAIRE
                #         # print("ligne:", ligne)
                #         obs[ligne] = 1389
                #
                #         # normalisation par rapport à la hauteur de l'ecran
                #         # obs[ligne] = np.interp(obs[ligne], [-SCREEN_HEIGHT, SCREEN_HEIGHT], [-1, 1])
                #         # normalisation par rapport à la hauteur de l'ecran
                #         # obs[ligne] = np.interp(obs[ligne], [-SCREEN_HEIGHT, SCREEN_HEIGHT], [-1, 1])
                #
                #         # INFO 2: Vitesse latérale de rapprochement à l'astéroide
                #         ligne += 1
                #         #print("ligne:", ligne)
                #         obs[ligne] = 0
                #         # obs[ligne] = np.interp(obs[ligne], [-VITESSE_VAISSEAUX_Y_MAX, VITESSE_VAISSEAUX_Y_MAX], [-1, 1])
                #
                #         # INFO 3: Distance en x à l'asteroide
                #         ligne += 1
                #         # print("ligne:", ligne)
                #         obs[ligne] = 1389
                #         # obs[ligne] = np.interp(obs[ligne], [LEFT_LIMIT, RIGHT_LIMIT], [-1, 1]) if np.interp(obs[ligne], [LEFT_LIMIT, RIGHT_LIMIT], [-1, 1]) > 0 else 0

                    # # Niveau de carburant de chaque vaisseau
                    # obs[NB_VAISSEAUX*NB_ASTEROIDES*NB_OBSERVATION_UNITAIRE + id_ship] = 0


            # Vitesse relative (constante)
            # obs[NB_VAISSEAUX * NB_ASTEROIDES * NB_OBSERVATION_UNITAIRE + NB_VAISSEAUX] = np.interp(VITESSE_VAISSEAUX, [-VITESSE_VAISSEAUX, VITESSE_VAISSEAUX], [-1, 1])

            print("\nObservation:", obs)

        return obs

    def render(self, mode="human"):
        """ Affichage de l'environnement """



        # Création de la fenetre d'affichage
        if self.viewer is None:
            pygame.init()
            pygame.display.set_caption("Affectameute")
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            clock = pygame.time.Clock()

            screen.fill((150,255,255))
            img_abeille = pygame.image.load('abeille-removebg-preview.png')

            # Dessiner les astéroïdes
            for asteroid in self.asteroides:
                img = pygame.transform.scale(img_abeille, (img_abeille.get_size()[0] / SCREEN_HEIGHT * 100, img_abeille.get_size()[1] / SCREEN_WIDTH * 100))
                screen.blit(img, (asteroid.center_x, asteroid.center_y))
            # Dessiner les vaisseaux
            for ship in self.ships:
                if ship.inactif == True:
                    img = pygame.image.load('frelon-removebg.png')
                    img = pygame.transform.scale(img, (
                        img.get_size()[0] / SCREEN_HEIGHT * 100, img.get_size()[1] / SCREEN_WIDTH * 100))
                    black_surface = pygame.Surface(img.get_size())
                    black_surface.fill((0, 0, 0))  # remplit la surface avec la couleur noire
                    img.blit(black_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                    screen.blit(img, (ship.center_x, ship.center_y))
                else:
                    img = pygame.image.load('frelon-removebg.png')
                    img = pygame.transform.scale(img, (
                    img.get_size()[0] / SCREEN_HEIGHT * 100, img.get_size()[1] / SCREEN_WIDTH * 100))
                    screen.blit(img, (ship.center_x, ship.center_y))


            font = pygame.font.Font(None, 26)
            text = font.render("Score : " + str(self.score) + " (Max = "+str(NB_ASTEROIDES)+ ")", True, (150, 0, 255))
            screen.blit(text, (10, 10))


            # Définir les paramètres de la flèche
            x = 20  # Position x de la flèche
            y = 200  # Position y de la flèche
            space = 20
            size = 20  # Taille de la flèche
            if self.act == 1:
                direction = 'up'
            elif self.act == 0:
                direction = 'down'
            else:
                direction = 'none'

            # recherche le vaisseau qui subit l'action actuelle
            for ship in self.ships:
                if ship.id == self.id_ship_act:
                    vaisseau = ship
                if ship == None:
                    print('ERREUR')

            frel = pygame.image.load('frelon-removebg.png')
            frel_h = frel.get_size()[0] / SCREEN_HEIGHT * 100
            frel_w = frel.get_size()[1] / SCREEN_WIDTH * 100
            # Définir les points de la flèche en fonction de la direction
            if direction == 'up':
                points = [(vaisseau.center_x+space, vaisseau.center_y -2* size),
                          (vaisseau.center_x + size / 2+space, vaisseau.center_y-space), (vaisseau.center_x - size / 2+space, vaisseau.center_y-space)]
                pygame.draw.polygon(screen, (255, 0, 0), points)
            elif direction == 'down':
                points = [(vaisseau.center_x+space, vaisseau.center_y +3*size),
                          (vaisseau.center_x + size / 2+space, vaisseau.center_y+2*space), (vaisseau.center_x - size / 2+space, vaisseau.center_y+2*space)]
                pygame.draw.polygon(screen, (255, 0, 0), points)
            else:
                pygame.draw.circle(surface=screen, color=(255, 0, 0), center=(x,y-50), radius=10)

            for ship in self.ships:
                font = pygame.font.Font(None, 26)
                text = font.render("Carburant KV " + str(ship.id) + " : " + str(ship.reservoir), True, (150, 0, 0))
                screen.blit(text, (10, ship.id * 60 if ship.id != 0 else 40))


            # # Affichage des directions sur les côtés
            # if direction == 'up':
            #     points = [(x, y + size), (x + size / 2, y), (x - size / 2, y)]
            #     pygame.draw.polygon(screen, (255, 0, 0), points)
            # elif direction == 'down':
            #     points = [(x, y - size-100), (x + size / 2, y-100), (x - size / 2, y-100)]
            #     pygame.draw.polygon(screen, (255, 0, 0), points)
            # else:
            #     pygame.draw.circle(surface=screen, color=(255, 0, 0), center=(x,y-50), radius=10)
            #
            # for ship in self.ships:
            #     font = pygame.font.Font(None, 26)
            #     text = font.render("Carburant KV " + str(ship.id) + " : " + str(ship.reservoir), True, (150, 0, 0))
            #     screen.blit(text, (10, ship.id*60 if ship.id!= 0 else 40))

            pygame.display.flip()
            clock.tick(60)


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




# --------------------------------
# Bloc de test de l'environnement
# --------------------------------
if __name__ == '__main__':
    mode = 'normal'
    if mode == 'normal':
        env = AffectaMeuteEnv()
        ones = 0
        count = 0
        observation = env.reset()
        done = False

        while not done:
            env.render()

            # Selection d'une action aléatoire
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
        if done:
            print("dernière observation : ", observation)




        # for episode in range(10):
        #     for t in range(1000):
        #         # Affichage de l'environnement
        #         env.render()
        #
        #         # Selection d'une action aléatoire
        #         action = env.action_space.sample()
        #
        #         observation, reward, done, info = env.step(action)
        #
        #         # if env.score==4:
        #         #     st()
        #
        #         count += 1
        #
        #
        #         #Traitement final
        #         if done:
        #             print("Episode terminé après {:.0f} itérations    SCORE: {:.0f}".format(t, env.score))
        #             print("dernière observation : ", observation)
        #             break


        time.sleep(1)
        env.close()