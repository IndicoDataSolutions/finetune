pipeline {
  agent any
  stages {
    stage('Build Docker Image') {
      steps {
        sh 'docker container rm -f finetune || true'
        sh './docker/build_docker.sh '
      }
    }
    stage('Start Docker Image') {
      steps {
        sh './docker/start_docker.sh'
        sh '''echo "Currently in $PWD"
ls'''
      }
    }
    stage('Run Tests ') {
      steps {
        sh 'docker ps'
        sh 'docker exec finetune nosetests'
      }
    }
    stage('Remove container') {
      steps {
        sh 'docker rm -f finetune'
      }
    }
  }
}